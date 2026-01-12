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
from cv_bridge import CvBridge

# --- PX4 & Custom Msgs ---
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode, VehicleCommand, VehicleAttitude, VehicleOdometry, VehicleStatus
from sensor_msgs.msg import Image
from qjwdds.msg import ImageDeviation

# --- LeRobot Imports ---
from lerobot.policies.act.modeling_act import ACTPolicy

class State(Enum):
    TAKEOFF = 1
    HOVER = 2
    GUIDANCE = 3  # ACT 模型接管

class ACTController(Node):
    def __init__(self):
        super().__init__('act_inference_node')

        # --- 1. 参数设置 (保留原逻辑) ---
        self.declare_parameter('takeoff_height', 40.0) 
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_yaw_rate', 0.1)
        self.declare_parameter('target_loss_timeout', 0.5)
        
        # 模型路径 (指向你训练输出的 best 或 last 文件夹)
        # 注意：这里假设你在 workspace 根目录下运行，或者根据实际情况修改绝对路径
        self.model_path = "/home/shuai/EndToEnd_FPV/LeRobot/outputs/drone_act_test0/checkpoints/last/pretrained_model"

        # --- 2. 加载 LeRobot ACT 模型 ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Loading ACT model from {self.model_path} on {self.device}...")
        
        try:
            self.policy = ACTPolicy.from_pretrained(self.model_path)
            self.policy.to(self.device)
            self.policy.eval()
            self.get_logger().info("ACT Policy loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load ACT model: {e}")
            raise e

        # --- 3. ROS 通信设置 ---
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()
        
        # 订阅
        self.create_subscription(Image, '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image', self.img_callback, qos_best_effort)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, qos_best_effort)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_best_effort)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_best_effort)
        self.create_subscription(ImageDeviation, '/camera/image_deviation', self.deviation_callback, qos_best_effort) # 安全检测

        # 发布
        self.pub_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.pub_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.pub_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # --- 4. 状态变量 ---
        self.current_state = State.TAKEOFF
        self.current_att = None
        self.current_pos = None
        self.current_yaw = 0.0
        self.arming_state = 0
        self.nav_state = 0
        
        self.base_altitude = None
        self.takeoff_yaw = None
        self.hover_setpoint = None
        
        self.last_valid_target_time = 0.0
        self.target_detected = False
        
        # 网络输出缓存
        self.net_vel_body = np.array([0.0, 0.0, 0.0])
        self.net_yaw_rate = 0.0
        
        self.offboard_setpoint_counter = 0
        self.is_armed_recorded = False

        # 50Hz 控制循环 (LeRobot ACT 通常不需要太高频，但保持 PX4 心跳需要至少 2Hz，这里用 20Hz 足够)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("ACT Controller Started. Waiting for Offboard...")

    # --- 回调函数 ---
    def status_callback(self, msg):
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def attitude_callback(self, msg):
        self.current_att = msg.q
        # 计算当前 Yaw (用于起飞锁定)
        q = msg.q
        siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.current_pos = msg.position

    def deviation_callback(self, msg):
        # 安全策略：只要有 deviation 消息且不是 NaN，说明识别到了红球
        if not math.isnan(msg.angle_x):
            self.last_valid_target_time = time.time()
            self.target_detected = True

    def img_callback(self, msg):
        # 只有在接管模式下才进行推理，节省资源
        if self.current_state != State.GUIDANCE:
            return
        
        if self.current_att is None:
            return

        try:
            # 1. 图像预处理
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # 转 RGB 并归一化到 [0, 1]
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(cv_img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device) # (1, 3, H, W)

            # 2. 构造 Dummy State (关键：维度必须是 6，与训练时一致)
            state_tensor = torch.zeros(1, 6).to(self.device)

            # 3. 构造 Observation 字典
            observation = {
                "observation.images.camera": img_tensor,
                "observation.state": state_tensor
            }

            # 4. 推理
            with torch.inference_mode():
                # select_action 自动处理 temporal ensembling 或返回单步动作
                action = self.policy.select_action(observation)
            
            # 5. 解析输出
            # action shape: (4,) -> [vx, vy, vz, yaw_rate] (Body Frame)
            raw_output = action.cpu().numpy().squeeze()
            
            # 6. 安全限幅
            max_spd = self.get_parameter('max_speed').value
            max_rate = self.get_parameter('max_yaw_rate').value
            
            self.net_vel_body = np.clip(raw_output[:3], -max_spd, max_spd)
            self.net_yaw_rate = np.clip(raw_output[3], -max_rate, max_rate)

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

    # --- 辅助函数 (保留原逻辑) ---
    def q_rotate(self, q, v):
        """ 机体系 -> NED 系 """
        w, x, y, z = q
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
        msg.param1 = param1; msg.param2 = param2
        msg.target_system = 1; msg.target_component = 1
        msg.source_system = 1; msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_vehicle_command.publish(msg)

    def publish_setpoint(self, position=None, velocity=None, yaw=None, yaw_rate=None):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # Position
        if position is not None:
            msg.position = [float(position[0]), float(position[1]), float(position[2])]
        else:
            msg.position = [float('nan'), float('nan'), float('nan')]
            
        # Velocity
        if velocity is not None:
            msg.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        else:
            msg.velocity = [float('nan'), float('nan'), float('nan')]
            
        # Yaw vs YawSpeed 互斥
        if yaw is not None:
            msg.yaw = float(yaw)
            msg.yawspeed = float('nan')
        elif yaw_rate is not None:
            msg.yaw = float('nan')
            msg.yawspeed = float(yaw_rate)
        else:
            msg.yaw = float('nan')
            msg.yawspeed = float('nan')

        self.pub_trajectory.publish(msg)

    # --- 主控制循环 ---
    def timer_callback(self):
        if self.current_pos is None or self.current_att is None:
            return

        # 1. 发布 Offboard 模式心跳
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        
        if self.current_state == State.GUIDANCE:
            # 接管模式：速度控制
            offboard_msg.position = False
            offboard_msg.velocity = True
        else:
            # 起飞/悬停模式：位置控制
            offboard_msg.position = True
            offboard_msg.velocity = False
            
        self.pub_offboard_mode.publish(offboard_msg)

        # 2. 初始解锁与起飞逻辑 (Sequence)
        if self.offboard_setpoint_counter < 100:
            self.offboard_setpoint_counter += 1
            # 预热：发送当前位置作为设定点
            self.publish_setpoint(position=self.current_pos, yaw=self.current_yaw)
            return

        if self.offboard_setpoint_counter == 100:
            # 尝试切换到 Offboard 模式
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        if self.offboard_setpoint_counter == 150 and not self.is_armed_recorded:
            # 解锁
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            
            # 记录起飞基准
            self.base_altitude = self.current_pos[2] # PX4 NED Z 是负的
            self.takeoff_yaw = self.current_yaw
            
            # 计算悬停目标点 (注意：NED Z 向上为负，所以起飞是减去高度)
            target_z = self.base_altitude - self.get_parameter('takeoff_height').value
            self.hover_setpoint = [self.current_pos[0], self.current_pos[1], target_z]
            
            self.is_armed_recorded = True
            self.get_logger().info(f"ARMED. Target Alt: {target_z:.2f}, Yaw Locked: {self.takeoff_yaw:.2f}")

        self.offboard_setpoint_counter += 1
        if not self.is_armed_recorded: return

        # 3. 状态机逻辑
        if self.current_state == State.TAKEOFF:
            # 执行起飞
            self.publish_setpoint(position=self.hover_setpoint, yaw=self.takeoff_yaw)
            
            # 检查是否到达高度 (误差小于 0.5m)
            z_err = abs(self.current_pos[2] - self.hover_setpoint[2])
            if z_err < 0.5:
                self.get_logger().info("Takeoff Reached -> HOVER")
                self.current_state = State.HOVER

        elif self.current_state == State.HOVER:
            # 悬停等待目标
            self.publish_setpoint(position=self.hover_setpoint, yaw=self.takeoff_yaw)
            
            timeout = self.get_parameter('target_loss_timeout').value
            is_target_fresh = (time.time() - self.last_valid_target_time) < timeout
            
            if is_target_fresh:
                self.get_logger().info("Target Detected -> GUIDANCE (ACT Model)")
                self.current_state = State.GUIDANCE

        elif self.current_state == State.GUIDANCE:
            # 安全检查：目标丢失处理
            timeout = self.get_parameter('target_loss_timeout').value
            if (time.time() - self.last_valid_target_time) > timeout:
                self.get_logger().warn("Target Lost -> HOVER")
                # 原地悬停：更新悬停点为当前位置
                self.hover_setpoint = list(self.current_pos)
                # 更新 yaw 为当前 yaw，防止回转
                self.takeoff_yaw = self.current_yaw 
                self.current_state = State.HOVER
                return

            # --- 执行模型控制 ---
            # 坐标转换：Body Velocity -> NED Velocity
            vel_ned = self.q_rotate(self.current_att, self.net_vel_body)
            
            # 发布控制：NED速度 + 角速度 (yaw_rate)
            self.publish_setpoint(velocity=vel_ned, yaw_rate=self.net_yaw_rate)
            
            # 打印日志
            if self.offboard_setpoint_counter % 20 == 0:
                self.get_logger().info(f"ACT Action -> Vel: {self.net_vel_body} | YawRate: {self.net_yaw_rate:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = ACTController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()