#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from enum import Enum
import os
import time
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from cv_bridge import CvBridge
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleCommand, VehicleAttitude, VehicleOdometry, VehicleStatus
from sensor_msgs.msg import Image
from qjwdds.msg import ImageDeviation

class State(Enum):
    TAKEOFF = 1
    HOVER = 2     # 定点悬停，等待目标
    GUIDANCE = 3  # 端到端神经网络接管

# --- 2. 神经网络定义 (必须与训练时完全一致) ---
class E2EPilotNet(nn.Module):
    def __init__(self, output_dim=4):
        super(E2EPilotNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5184, 100), nn.ReLU(),
            nn.Dropout(0.3),     # 必须保留，匹配权重文件结构
            nn.Linear(100, 50),   nn.ReLU(),
            nn.Linear(50, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. 控制节点 ---
class E2EStateMachine(Node):
    def __init__(self):
        super().__init__('e2e_state_machine')

        # --- 参数设置 ---
        self.declare_parameter('takeoff_height', 10.0) # 相对解锁高度 10m
        self.declare_parameter('max_speed', 3.0)       # 神经网络最大速度限制
        self.declare_parameter('target_loss_timeout', 0.5) # 目标丢失超时时间

        # --- 路径与模型加载 ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'e2e_model.pth')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = E2EPilotNet().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info(f"Model loaded successfully: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            # exit(1) # 实际部署建议取消注释

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- QoS 设置 ---
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- 订阅与发布 ---
        self.bridge = CvBridge()
        
        # 订阅传感器 default baylands
        self.create_subscription(Image, '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image', self.img_callback, qos_best_effort)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, qos_best_effort)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_best_effort)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_best_effort)
        
        # 触发器订阅 (红球检测结果)
        self.create_subscription(ImageDeviation, '/camera/image_deviation', self.deviation_callback, qos_best_effort)

        # 发布控制
        self.pub_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.pub_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.pub_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # --- 状态变量 ---
        self.current_state = State.TAKEOFF
        self.current_att = None # [w, x, y, z]
        self.current_pos = None # [x, y, z] NED
        self.current_yaw = 0.0
        self.arming_state = 0
        self.nav_state = 0
        
        # 关键锚点数据
        self.base_altitude = None   # 地面高度 (NED Z)
        self.takeoff_yaw = None     # 起飞时的锁定航向
        self.hover_setpoint = None  # 悬停目标点 [x, y, z]
        
        # 目标检测状态
        self.last_valid_target_time = 0.0
        self.target_detected = False
        
        # 神经网络输出缓存
        self.net_vel_body = np.array([0.0, 0.0, 0.0])
        self.net_yaw_cmd = 0.0
        
        # 流程控制
        self.offboard_setpoint_counter = 0
        self.is_armed_recorded = False

        # 主循环 20Hz
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("E2E State Machine Node Started")

    # --- 回调函数 ---
    def status_callback(self, msg):
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def attitude_callback(self, msg):
        self.current_att = msg.q
        # 计算当前 Yaw (用于记录起飞航向)
        q = msg.q
        siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.current_pos = msg.position # NED

    def deviation_callback(self, msg):
        # 只要收到且数据不是NaN，就视为有效目标
        if not math.isnan(msg.angle_x):
            self.last_valid_target_time = time.time()
            self.target_detected = True
        else:
            # 这一帧没看到
            pass

    def img_callback(self, msg):
        # 仅在 GUIDANCE 模式下进行推理，节省资源
        if self.current_state != State.GUIDANCE:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            input_tensor = self.transform(cv_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = output.cpu().numpy()[0] # [vx, vy, vz, yaw]
            
            # 限制速度幅度
            max_spd = self.get_parameter('max_speed').value
            self.net_vel_body = np.clip(prediction[:3], -max_spd, max_spd)
            self.net_yaw_cmd = prediction[3] 

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

    # --- 辅助函数 ---
    def q_rotate(self, q, v):
        """ 将机体坐标系向量旋转到 NED 坐标系 """
        # v_ned = R(q) * v_body
        w, x, y, z = q
        # 旋转矩阵公式
        r00 = 1 - 2*y*y - 2*z*z
        r01 = 2*x*y - 2*w*z
        r02 = 2*x*z + 2*w*y
        r10 = 2*x*y + 2*w*z
        r11 = 1 - 2*x*x - 2*z*z
        r12 = 2*y*z - 2*w*x
        r20 = 2*x*z - 2*w*y
        r21 = 2*y*z + 2*w*x
        r22 = 1 - 2*x*x - 2*y*y
        
        vx = r00*v[0] + r01*v[1] + r02*v[2]
        vy = r10*v[0] + r11*v[1] + r12*v[2]
        vz = r20*v[0] + r21*v[1] + r22*v[2]
        return np.array([vx, vy, vz])

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_vehicle_command.publish(msg)

    # --- 主逻辑 ---
    def timer_callback(self):
        # 1. 基础数据检查
        if self.current_pos is None or self.current_att is None:
            if self.offboard_setpoint_counter % 20 == 0:
                self.get_logger().info("Waiting for Odom/Attitude...")
            return

        # 2. 发布 Offboard 模式心跳
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        
        # 根据状态决定控制模式
        if self.current_state == State.GUIDANCE:
            offboard_msg.position = False
            offboard_msg.velocity = True  # GUIDANCE 用速度控制
        else:
            offboard_msg.position = True  # TAKEOFF/HOVER 用位置控制
            offboard_msg.velocity = False
            
        self.pub_offboard_mode.publish(offboard_msg)

        # 3. 初始化序列 (Offboard -> Arm -> Record Base Info)
        if self.offboard_setpoint_counter < 100:
            self.offboard_setpoint_counter += 1
            # 发送当前位置作为待命
            self.publish_setpoint(position=self.current_pos, yaw=self.current_yaw)
            return

        # 切换 Offboard
        if self.offboard_setpoint_counter == 100:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.get_logger().info(">>> REQUESTING OFFBOARD MODE")

        # 解锁并记录基准信息
        if self.offboard_setpoint_counter == 150 and not self.is_armed_recorded:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            
            # [关键] 记录解锁时的状态
            self.base_altitude = self.current_pos[2]    # 记录地面高度 (NED Z)
            self.takeoff_yaw = self.current_yaw         # 记录当前航向，之后全程锁定
            
            # 计算起飞目标点 (当前X, 当前Y, 地面Z - 10m)
            target_z = self.base_altitude - self.get_parameter('takeoff_height').value
            self.hover_setpoint = [self.current_pos[0], self.current_pos[1], target_z]
            
            self.is_armed_recorded = True
            self.get_logger().info(f">>> ARMED. Base Alt: {self.base_altitude:.2f}, Target Alt: {target_z:.2f}, Yaw Locked: {self.takeoff_yaw:.2f}")

        self.offboard_setpoint_counter += 1
        if not self.is_armed_recorded: return # 等待解锁完成

        # 4. 状态机逻辑
        
        # --- STATE: TAKEOFF (起飞) ---
        if self.current_state == State.TAKEOFF:
            # 策略：位置控制，去往 (Hover_X, Hover_Y, Target_Z)
            self.publish_setpoint(position=self.hover_setpoint, yaw=self.takeoff_yaw)
            
            # 检查是否到达高度 (误差 < 0.5m)
            z_err = abs(self.current_pos[2] - self.hover_setpoint[2])
            if z_err < 0.5:
                self.get_logger().info(">>> Takeoff Reached. Switching to HOVER.")
                self.current_state = State.HOVER

        # --- STATE: HOVER (悬停) ---
        elif self.current_state == State.HOVER:
            # 策略：位置控制，死守 Hover Point，锁定 Yaw
            self.publish_setpoint(position=self.hover_setpoint, yaw=self.takeoff_yaw)
            
            # 触发机制：检查是否有目标
            timeout = self.get_parameter('target_loss_timeout').value
            is_target_fresh = (time.time() - self.last_valid_target_time) < timeout
            
            if is_target_fresh:
                self.get_logger().info(">>> Target Detected! Switching to GUIDANCE (Neural Net).")
                self.current_state = State.GUIDANCE

        # --- STATE: GUIDANCE (端到端接管) ---
        elif self.current_state == State.GUIDANCE:
            # 丢失检查
            timeout = self.get_parameter('target_loss_timeout').value
            if (time.time() - self.last_valid_target_time) > timeout:
                self.get_logger().warn(">>> Target Lost! Switching back to HOVER.")
                
                # [关键] 切回悬停时，更新悬停点为当前位置，防止无人机猛烈回弹
                self.hover_setpoint = list(self.current_pos)
                # 保持高度不低于 2m，防止掉下来
                # self.hover_setpoint[2] = min(self.hover_setpoint[2], self.base_altitude - 2.0)
                
                self.current_state = State.HOVER
                return

            # E2E 控制执行
            # 1. 转换速度：Body Frame -> NED Frame
            vel_ned = self.q_rotate(self.current_att, self.net_vel_body)
            
            # 2. 发布速度指令 (注意：Position 设为 NaN)
            # 如果你的模型输出的是绝对Yaw，用 net_yaw_cmd；如果是YawRate，逻辑不同
            self.publish_setpoint(velocity=vel_ned, yaw=self.net_yaw_cmd)
            
            # Debug log
            if self.offboard_setpoint_counter % 20 == 0:
                self.get_logger().info(f"E2E Body Vel: {self.net_vel_body} | Yaw: {self.net_yaw_cmd:.2f}")

    def publish_setpoint(self, position=None, velocity=None, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # 填充位置 (如果为 None 则填 NaN)
        if position is not None:
            msg.position = [float(position[0]), float(position[1]), float(position[2])]
        else:
            msg.position = [float('nan'), float('nan'), float('nan')]
            
        # 填充速度
        if velocity is not None:
            msg.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        else:
            msg.velocity = [float('nan'), float('nan'), float('nan')]
            
        msg.yaw = float(yaw)
        self.pub_trajectory.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = E2EStateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()