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
    HOVER = 2
    GUIDANCE = 3  # 神经网络接管 (速度 + 角速度控制)

# --- 神经网络定义 ---
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
            nn.Dropout(0.3),
            nn.Linear(100, 50),   nn.ReLU(),
            nn.Linear(50, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 控制节点 ---
class E2EStateMachine(Node):
    def __init__(self):
        super().__init__('vision_control_e2e')

        # --- 参数设置 ---
        self.declare_parameter('takeoff_height', 10.0) 
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('max_yaw_rate', 0.5)    # 最大角速度限制
        self.declare_parameter('target_loss_timeout', 0.5) 

        # --- 模型加载 ---
        home_dir = os.path.expanduser('~')
        model_path = os.path.join(home_dir, 'sh_ws/src/qjwddspy/qjwddspy/models/default_e2e_model_3.pth')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = E2EPilotNet().to(self.device)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info(f"Model loaded: {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- QoS ---
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
        self.create_subscription(ImageDeviation, '/camera/image_deviation', self.deviation_callback, qos_best_effort)

        self.pub_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.pub_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.pub_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # --- 状态变量 ---
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
        
        self.net_vel_body = np.array([0.0, 0.0, 0.0])
        self.net_yaw_rate = 0.0
        
        self.offboard_setpoint_counter = 0
        self.is_armed_recorded = False

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("E2E Control Started")

    def status_callback(self, msg):
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def attitude_callback(self, msg):
        self.current_att = msg.q
        q = msg.q
        siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        self.current_yaw = np.arctan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.current_pos = msg.position

    def deviation_callback(self, msg):
        if not math.isnan(msg.angle_x):
            self.last_valid_target_time = time.time()
            self.target_detected = True
        # else:
        #     # 这一帧没看到
        #     pass
    def img_callback(self, msg):
        if self.current_state != State.GUIDANCE:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(cv_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = output.cpu().numpy()[0] # [vx, vy, vz, yaw_rate]
            
            max_spd = self.get_parameter('max_speed').value
            max_rate = self.get_parameter('max_yaw_rate').value
            
            # 限制线速度
            self.net_vel_body = np.clip(prediction[:3], -max_spd, max_spd)
            
            # 限制角速度
            self.net_yaw_rate = np.clip(prediction[3], -max_rate, max_rate)

        except Exception as e:
            self.get_logger().error(f"Inference Error: {e}")

    def q_rotate(self, q, v):
        """ 将机体坐标系向量旋转到 NED 坐标系 """
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

    def timer_callback(self):
        if self.current_pos is None or self.current_att is None:
            return

        # 发布 Offboard 控制模式
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        
        if self.current_state == State.GUIDANCE:
            offboard_msg.position = False
            offboard_msg.velocity = True
        else:
            offboard_msg.position = True
            offboard_msg.velocity = False
            
        self.pub_offboard_mode.publish(offboard_msg)

        # 初始化
        if self.offboard_setpoint_counter < 100:
            self.offboard_setpoint_counter += 1
            self.publish_setpoint(position=self.current_pos, yaw=self.current_yaw) # 待命使用绝对yaw
            return

        if self.offboard_setpoint_counter == 100:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

        if self.offboard_setpoint_counter == 150 and not self.is_armed_recorded:
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            # 解锁时的状态
            self.base_altitude = self.current_pos[2] # (NED Z)
            self.takeoff_yaw = self.current_yaw
            target_z = self.base_altitude - self.get_parameter('takeoff_height').value
            self.hover_setpoint = [self.current_pos[0], self.current_pos[1], target_z]
            self.is_armed_recorded = True
            self.get_logger().info(f"ARMED. Yaw Locked: {self.takeoff_yaw:.2f}")

        self.offboard_setpoint_counter += 1
        if not self.is_armed_recorded: return

        # 状态机逻辑
        if self.current_state == State.TAKEOFF:
            self.publish_setpoint(position=self.hover_setpoint, yaw=self.takeoff_yaw) # [位置+绝对Yaw]
            
            z_err = abs(self.current_pos[2] - self.hover_setpoint[2])
            if z_err < 0.5:
                self.get_logger().info("Takeoff Reached -> HOVER")
                self.current_state = State.HOVER

        elif self.current_state == State.HOVER:
            self.publish_setpoint(position=self.hover_setpoint, yaw=self.takeoff_yaw) # [位置+绝对Yaw]
            
            timeout = self.get_parameter('target_loss_timeout').value
            is_target_fresh = (time.time() - self.last_valid_target_time) < timeout
            if is_target_fresh:
                self.get_logger().info("Target Detected -> GUIDANCE")
                self.current_state = State.GUIDANCE

        elif self.current_state == State.GUIDANCE:
            timeout = self.get_parameter('target_loss_timeout').value
            if (time.time() - self.last_valid_target_time) > timeout:
                self.get_logger().warn("Target Lost -> HOVER")
                self.hover_setpoint = list(self.current_pos)
                # 切回 HOVER 时，更新 takeoff_yaw 为当前朝向，防止猛烈回转
                self.takeoff_yaw = self.current_yaw 
                self.current_state = State.HOVER
                return

            # E2E 控制 (速度 + 角速度)
            vel_ned = self.q_rotate(self.current_att, self.net_vel_body)
            
            # 使用 yaw_rate 参数
            self.publish_setpoint(velocity=vel_ned, yaw_rate=self.net_yaw_rate)
            
            if self.offboard_setpoint_counter % 20 == 0:
                self.get_logger().info(f"Vel: {self.net_vel_body} | Rate: {self.net_yaw_rate:.2f}")

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