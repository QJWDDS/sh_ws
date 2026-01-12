#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from px4_msgs.msg import TrajectorySetpoint, VehicleAttitude
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import csv
import time
import math

class DataRecorder(Node):
    def __init__(self):
        super().__init__('e2e_data_recorder')

        # --- 参数设置 ---
        self.declare_parameter('save_dir', os.path.expanduser('~/sh_ws/document/default_data/e2eindependent'))
        self.save_dir = self.get_parameter('save_dir').value

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.save_dir, timestamp)
        self.img_dir = os.path.join(self.session_dir, 'images')
        os.makedirs(self.img_dir, exist_ok=True)
        
        # 初始化 CSV 文件
        self.csv_path = os.path.join(self.session_dir, 'data.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['img_name', 'v_body_x', 'v_body_y', 'v_body_z', 'yaw_rate_cmd'])

        # --- QoS 设置 ---
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- 订阅话题 ---
        self.create_subscription(
            Image,
            '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image',
            self.img_callback,
            qos_sensor
        )
        
        # 控制指令
        self.create_subscription(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            self.cmd_callback,
            qos_profile
        )

        # 姿态
        self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile
        )

        self.bridge = CvBridge()

        self.latest_traj = None
        self.latest_att = None
        self.count = 0
        
        self.get_logger().info(f"Data Recorder Started. Saving to: {self.session_dir}")

    def cmd_callback(self, msg):
        self.latest_traj = msg

    def attitude_callback(self, msg):
        self.latest_att = msg

    def q_inv(self, q):
        """四元数共轭"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def q_mult(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    def rotate_vector(self, q, v):
        """使用四元数旋转向量 v_new = q * v * q_inv"""
        # 将向量转换为纯虚四元数 [0, v]
        v_q = np.array([0.0, v[0], v[1], v[2]])
        q_inv = self.q_inv(q)
        temp = self.q_mult(q, v_q)
        result = self.q_mult(temp, q_inv)
        return result[1:] 

    def ned_to_body(self, q_ned_to_body, v_ned):
        q_inv = self.q_inv(q_ned_to_body)
        return self.rotate_vector(q_inv, v_ned)

    def img_callback(self, msg):
        # 完整性
        if self.latest_traj is None or self.latest_att is None:
            return

        # 是否在 GUIDANCE 模式

        vx_ned = self.latest_traj.velocity[0]
        vy_ned = self.latest_traj.velocity[1]
        vz_ned = self.latest_traj.velocity[2]
        desired_yawspeed = self.latest_traj.yawspeed

        if np.isnan(vx_ned):
            return

        try:
            # 图像处理
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            q_current = np.array(self.latest_att.q)
            v_ned_vec = np.array([vx_ned, vy_ned, vz_ned])
            # 计算机体速度
            v_body = self.ned_to_body(q_current, v_ned_vec)

            # 保存数据
            filename = f"{self.count:06d}.jpg"
            img_save_path = os.path.join(self.img_dir, filename)
            
            cv2.imwrite(img_save_path, cv_image)          
            # 写入 CSV: [文件名, 机体前向v, 机体右向v, 机体垂直v, 期望yaw]
            self.csv_writer.writerow([
                filename, 
                f"{v_body[0]:.4f}", 
                f"{v_body[1]:.4f}", 
                f"{v_body[2]:.4f}",
                f"{desired_yawspeed:.4f}"
            ])
            
            self.count += 1
            if self.count % 100 == 0:
                self.get_logger().info(f"Recorded {self.count} frames...")

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()