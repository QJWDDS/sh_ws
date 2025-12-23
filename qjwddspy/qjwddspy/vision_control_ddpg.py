import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from collections import deque
import torch
import torch.nn as nn
import numpy as np
import math
import os
import time
import datetime

# --- PX4 & Custom Msgs ---
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleAttitude, VehicleOdometry
try:
    from qjwdds.msg import ImageDeviation
except ImportError:
    print("Warning: qjwdds.msg not found. Ensure the package is sourced.")
    ImageDeviation = None

# DRL Actor 网络结构
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = torch.FloatTensor(max_action)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.max_action.to(x.device) * self.tanh(self.l3(x))


# 视觉控制节点
class VisionControlDRL(Node):
    def __init__(self):
        super().__init__('vision_control_ddpg')

        home_dir = os.path.expanduser('~')
        default_log_dir = os.path.join(home_dir, 'sh_ws/document/datelogs/')
        default_model_path = os.path.join(home_dir, 'sh_ws/src/qjwddspy/qjwddspy/uav_actor.pth')

        # --- 参数配置 ---
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('takeoff_rel_alt', 5.0)

        self.declare_parameter('max_speed', 5.0)            
        self.declare_parameter('max_yaw_rate', 10.0) #度          
        self.declare_parameter('target_loss_timeout', 2.0)
        self.declare_parameter('log_dir', default_log_dir)
        self.declare_parameter('vertical_gain', 1.0)
        
        # 减小平滑因子
        self.declare_parameter('smooth_factor', 0.05)       
        self.declare_parameter('deadzone_deg', 3.0)

        self.filter_window = 5 
        self.angle_x_history = deque(maxlen=self.filter_window)
        self.angle_y_history = deque(maxlen=self.filter_window)

        self.max_accel_xy = 1.0  
        self.max_accel_z = 1.0
        self.last_cmd_vel = np.zeros(3)

        # 读取参数
        self.model_path = self.get_parameter('model_path').value
        self.takeoff_h = self.get_parameter('takeoff_rel_alt').value
        self.max_v = self.get_parameter('max_speed').value
        self.max_omega = np.deg2rad(self.get_parameter('max_yaw_rate').value)
        self.loss_timeout = self.get_parameter('target_loss_timeout').value
        self.log_dir = self.get_parameter('log_dir').value
        self.v_gain = self.get_parameter('vertical_gain').value
        self.smooth_factor = self.get_parameter('smooth_factor').value
        self.deadzone_rad = np.deg2rad(self.get_parameter('deadzone_deg').value)

        # --- 状态变量 ---
        self.state = "TAKEOFF"
        self.arm_pos = None
        self.arm_yaw = None
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_att_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.current_yaw = 0.0
        
        self.has_target = False
        self.last_target_time = 0.0
        self.angle_x = 0.0 
        self.angle_y = 0.0 
        
        self.target_yaw = 0.0
        self.offboard_setpoint_counter = 0

        # 平滑滤波变量
        self.last_action_v = 0.0
        self.last_action_omega = 0.0

        # --- 加载模型 ---
        self.device = torch.device("cpu")
        train_max_action = np.array([5.0, np.deg2rad(60.0)]) 
        self.actor = Actor(4, 2, train_max_action).to(self.device)
        
        if os.path.exists(self.model_path):
            try:
                self.actor.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.actor.eval()
                self.get_logger().info(f"Loaded Actor model from {self.model_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load model: {e}")
        else:
            self.get_logger().error(f"Model file not found: {self.model_path}")

        # --- ROS 通讯 ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_odom = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_cb, qos_profile)
        self.sub_att = self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.att_cb, qos_profile)
        if ImageDeviation:
            self.sub_img = self.create_subscription(ImageDeviation, '/camera/image_deviation', self.img_cb, qos_profile)
        
        self.pub_offboard = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.pub_cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        self.init_logger()
        self.timer = self.create_timer(0.02, self.control_loop)

    def init_logger(self):
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except PermissionError:
                self.get_logger().error(f"Permission denied creating log dir: {self.log_dir}. Changing to /tmp/")
                self.log_dir = "/tmp/"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(self.log_dir, f"uav_control_{timestamp}.txt")
        try:
            with open(self.log_file_path, 'w') as f:
                f.write("time,pos_x,pos_y,pos_z,att_w,att_x,att_y,att_z,yaw,angle_x,angle_y,state\n")
            self.get_logger().info(f"Logging to {self.log_file_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to init log file: {e}")

    def log_state(self):
        try:
            with open(self.log_file_path, 'a') as f:
                now = datetime.datetime.now().strftime("%H:%M:%S.%f")
                f.write(f"{now},{self.current_pos[0]:.3f},{self.current_pos[1]:.3f},{self.current_pos[2]:.3f},"
                        f"{self.current_att_q[0]:.3f},{self.current_att_q[1]:.3f},{self.current_att_q[2]:.3f},{self.current_att_q[3]:.3f},"
                        f"{self.current_yaw:.3f},{self.angle_x:.2f},{self.angle_y:.2f},{self.state}\n")
        except Exception:
            pass

    def q2yaw(self, q):
        return np.arctan2(2.0*(q[1]*q[2] + q[0]*q[3]), q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2)

    def body_to_ground(self, u_body, q):
        w, x, y, z = q
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
            [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]
        ])
        return R @ u_body

    def odom_cb(self, msg):
        self.current_pos = np.array(msg.position)
        self.current_vel = np.array(msg.velocity)

    def att_cb(self, msg):
        self.current_att_q = np.array(msg.q)
        self.current_yaw = self.q2yaw(self.current_att_q)

    def img_cb(self, msg):
        if not math.isnan(msg.angle_x) and not math.isnan(msg.angle_y):
            self.angle_x_history.append(msg.angle_x)
            self.angle_y_history.append(msg.angle_y)
            self.last_target_time = time.time()
            self.has_target = True

    def get_filtered_angles(self):
        if len(self.angle_x_history) == 0:
            return 0.0, 0.0
        avg_x = sum(self.angle_x_history) / len(self.angle_x_history)
        avg_y = sum(self.angle_y_history) / len(self.angle_y_history)
        return avg_x, avg_y


    # 主控制循环
    def control_loop(self):
        self.publish_offboard_mode()
        self.log_state()

        if np.linalg.norm(self.current_pos) < 0.001 and self.offboard_setpoint_counter == 0:
            return

        now = time.time()
        dt_img = now - self.last_target_time
        
        # 目标丢失检测
        is_target_lost = dt_img > self.loss_timeout or not self.has_target
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        traj_msg.position = [float('nan')]*3
        traj_msg.velocity = [float('nan')]*3
        traj_msg.acceleration = [float('nan')]*3

        if self.state == "TAKEOFF":
            if self.offboard_setpoint_counter == 10:
                self.arm_pos = self.current_pos.copy()
                self.arm_yaw = self.current_yaw
                self.target_yaw = self.arm_yaw 
                self.engage_offboard_and_arm()
            self.offboard_setpoint_counter += 1

            if self.arm_pos is not None:
                target_z = self.arm_pos[2] - self.takeoff_h
                traj_msg.position = [float(self.arm_pos[0]), float(self.arm_pos[1]), float(target_z)]
                traj_msg.yaw = float(self.arm_yaw)
                if abs(self.current_pos[2] - target_z) < 0.5:
                    self.get_logger().info("Takeoff Complete. Switching to HOVER.")
                    self.state = "HOVER"

        elif self.state == "HOVER":
            traj_msg.velocity = [0.0, 0.0, 0.0]
            traj_msg.yaw = float(self.current_yaw)
            self.last_action_v = 0.0
            self.last_action_omega = 0.0
            self.last_cmd_vel = np.zeros(3)
            
            if not is_target_lost:
                self.get_logger().info("Target detected. Switching to GUIDANCE.")
                self.state = "GUIDANCE"
                self.target_yaw = self.current_yaw
                self.angle_x_history.clear()
                self.angle_y_history.clear()

        elif self.state == "GUIDANCE":
            if is_target_lost:
                self.get_logger().warn("Target lost. Switching to HOVER.")
                self.state = "HOVER"
                return 

            filt_angle_x, filt_angle_y = self.get_filtered_angles()
            
            obs_angle = -np.radians(filt_angle_x) 
            v_horiz = np.linalg.norm(self.current_vel[:2])
            obs = torch.tensor([v_horiz, 0.0, obs_angle, 1.0], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actor(obs).cpu().numpy().flatten()

            raw_v = np.clip(action[0], 0.0, self.max_v)
            raw_omega = np.clip(action[1], -self.max_omega, self.max_omega)

            error_magnitude = abs(filt_angle_x)
            if error_magnitude > 10.0:
                curr_smooth = 0.3  
            elif error_magnitude > 5.0:
                curr_smooth = 0.15 
            else:
                curr_smooth = 0.1 # 平滑 
            
            self.last_action_v = (1 - curr_smooth) * self.last_action_v + curr_smooth * raw_v
            self.last_action_omega = (1 - curr_smooth) * self.last_action_omega + curr_smooth * raw_omega

            dz_deg = np.rad2deg(self.deadzone_rad)
            if abs(filt_angle_x) < dz_deg:
                ratio = filt_angle_x / dz_deg
                self.last_action_omega *= (ratio ** 2) 
            
            # 航向更新
            # 注意：dt 约为 0.02s，需与 timer 频率一致
            self.target_yaw += (-self.last_action_omega) * 0.02

            target_v_x = self.last_action_v * np.cos(self.target_yaw)
            target_v_y = self.last_action_v * np.sin(self.target_yaw)

            ax = np.radians(filt_angle_x)
            ay = np.radians(filt_angle_y)
            u_b = np.array([math.cos(ay)*math.cos(ax), math.cos(ay)*math.sin(ax), math.sin(ay)])
            u_ned = self.body_to_ground(u_b, self.current_att_q)
            u_horiz_norm = np.linalg.norm(u_ned[:2])
            
            v_z_geom = 0.0
            if u_horiz_norm > 0.1:
                v_z_geom = self.last_action_v * (u_ned[2] / u_horiz_norm)
            
            kp_z = 2.0 
            v_z_p = kp_z * u_ned[2] * self.max_v 
            weight_geom = 0.7
            weight_p = 0.3
            target_v_z = weight_geom * v_z_geom + weight_p * v_z_p

            # 高度保护与限幅
            min_z = self.arm_pos[2] - 20.0
            max_z = self.arm_pos[2] - 1.0
            pred_z = self.current_pos[2] + target_v_z * 0.5
            if pred_z < min_z and target_v_z < 0: target_v_z = 0.0
            if pred_z > max_z and target_v_z > 0: target_v_z = 0.0
            target_v_z = np.clip(target_v_z, -1.5, 1.5) 

            dt = 0.02

            curr_v_xy = np.array([target_v_x, target_v_y])
            last_v_xy = self.last_cmd_vel[:2]
            delta_v_xy = curr_v_xy - last_v_xy
            max_delta_xy = self.max_accel_xy * dt
            
            if np.linalg.norm(delta_v_xy) > max_delta_xy:
                delta_v_xy = max_delta_xy * (delta_v_xy / np.linalg.norm(delta_v_xy))
            
            final_v_xy = last_v_xy + delta_v_xy
            
            delta_vz = target_v_z - self.last_cmd_vel[2]
            delta_vz = np.clip(delta_vz, -self.max_accel_z * dt, self.max_accel_z * dt)
            final_vz = self.last_cmd_vel[2] + delta_vz

            self.last_cmd_vel = np.array([final_v_xy[0], final_v_xy[1], final_vz])

            traj_msg.velocity = [float(final_v_xy[0]), float(final_v_xy[1]), float(final_vz)]
            traj_msg.yaw = float(self.target_yaw)

        self.pub_traj.publish(traj_msg)


    # 命令发送
    def publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = (self.state == "TAKEOFF")
        msg.velocity = (self.state != "TAKEOFF")
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.pub_offboard.publish(msg)

    def engage_offboard_and_arm(self):
        self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        time.sleep(0.1)
        self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arming and switching to Offboard mode.")

    def send_command(self, cmd, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = cmd
        msg.param1 = float(p1)
        msg.param2 = float(p2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.pub_cmd.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisionControlDRL()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()