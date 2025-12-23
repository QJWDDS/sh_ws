#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from qjwdds.msg import ImageDeviation
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
import time
import logging
from typing import Tuple, Optional, Union
from cv_bridge import CvBridge, CvBridgeError

class ImageDectPublisher(Node):
    def __init__(self):
        super().__init__('gazebo_image_dec_pub')
        
        self.get_logger().set_level(logging.DEBUG)
        home_dir = os.path.expanduser('~')
        
        # 参数声明
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 30.0),
                ('calibration_file', os.path.join(home_dir, 'sh_ws/src/qjwddspy/qjwddspy/usb_ca_0515.yaml')),
                ('video_save_dir', os.path.join(home_dir, 'sh_ws/document/video')),
                ('recording_fps', 15.0),
                ('min_object_area', 300),
                ('hsv_lower', [0, 100, 100]), 
                ('hsv_upper', [10, 255, 255]),
                ('circularity_threshold', 0.6),
                ('video_output', True),
                ('image_topic', '/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image')
            ]
        )
        
        self.hsv_lower = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
        self.hsv_upper = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)
        self.min_area = self.get_parameter('min_object_area').value
        self.circularity_threshold = self.get_parameter('circularity_threshold').value
        self.video_output = self.get_parameter('video_output').value
        self.image_topic = self.get_parameter('image_topic').value
        
        self.video_save_dir = self.get_parameter('video_save_dir').value
        os.makedirs(self.video_save_dir, exist_ok=True)

        self.publisher = self.create_publisher(ImageDeviation, '/camera/image_deviation', 10)
        self.subscriber = self.create_subscription(
            Image, 
            self.image_topic, 
            self.image_callback, 
            10
        )
        
        self.bridge = CvBridge()
        
        # 相机标定参数
        self.K = None
        self.D = None
        self.load_calibration()
        
        # 视频录制
        self.video_writer = None
        self.is_recording = False
        
        # 状态变量
        self.frame_count = 0
        self.last_detection_time = time.time()
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.target_position = None
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.err_u = 0.0
        self.err_v = 0.0

    def load_calibration(self):
        """加载相机标定参数"""
        calibration_file = self.get_parameter('calibration_file').value
        if not os.path.isfile(calibration_file):
            self.get_logger().warn(f"标定文件未找到: {calibration_file}，角度计算将不可用")
            return

        try:
            fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
            if fs.isOpened():
                self.K = fs.getNode("K").mat()
                self.D = fs.getNode("D").mat()
                fs.release()
                self.get_logger().info("标定参数加载成功")
            else:
                self.get_logger().error("无法打开标定文件")
        except Exception as e:
            self.get_logger().error(f"加载标定文件异常: {e}")

    def start_recording(self, frame):
        if self.is_recording: return
            
        height, width = frame.shape[:2]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.video_save_dir, f"gz_ball_{timestamp}.mkv")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        self.video_writer = cv2.VideoWriter(
            video_path, fourcc, 
            self.get_parameter('recording_fps').value, 
            (width, height)
        )
        
        if self.video_writer.isOpened():
            self.is_recording = True
            self.get_logger().info(f"开始录像: {video_path}")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            self.get_logger().info("录像停止")

    def detect_spherical_object(self, frame) -> Optional[tuple]:
        if frame is None: return None
            
        try:
            # 转换颜色空间 BGR -> HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

            if self.hsv_lower[0] < 10:
                lower_red2 = np.array([170, self.hsv_lower[1], self.hsv_lower[2]])
                upper_red2 = np.array([180, self.hsv_upper[1], self.hsv_upper[2]])
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask, mask2)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_result = None
            max_circularity = 0.0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area: continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0: continue

                circularity = 4 * np.pi * area / (perimeter ** 2)

                if circularity > self.circularity_threshold and circularity > max_circularity:
                    max_circularity = circularity
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        best_result = (cX, cY, contour, circularity)

            return best_result

        except Exception as e:
            self.get_logger().error(f"检测算法出错: {e}")
            return None

    def calculate_angles(self, u, v):
        if self.K is None: return 0.0, 0.0
        fx, fy = self.K[0,0], self.K[1,1]
        ang_x = np.degrees(np.arctan(u / fx))
        ang_y = np.degrees(np.arctan(v / fy))
        return float(ang_x), float(ang_y)

    def image_callback(self, msg):
        # FPS 计算
        self.frame_count += 1
        if time.time() - self.last_fps_time > 2.0:
            self.current_fps = self.frame_count / (time.time() - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = time.time()

        try:
            # OpenCV 默认使用 BGR
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        if self.video_output and not self.is_recording:
            self.start_recording(frame)

        detect_result = self.detect_spherical_object(frame)
        
        msg_dev = ImageDeviation()
        
        if detect_result:
            cX, cY, contour, circularity = detect_result
            self.target_position = (cX, cY)

            h, w = frame.shape[:2]
            err_u = cX - w // 2
            err_v = cY - h // 2

            self.angle_x, self.angle_y = self.calculate_angles(err_u, err_v)
            self.err_u, self.err_v = float(err_u), float(err_v)

            msg_dev.err_u = self.err_u
            msg_dev.err_v = self.err_v
            msg_dev.angle_x = self.angle_x
            msg_dev.angle_y = self.angle_y

            self.last_detection_time = time.time()

            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Circ: {circularity:.2f}", (cX+10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:

            msg_dev.err_u = float('nan')
            msg_dev.err_v = float('nan')
            msg_dev.angle_x = float('nan')
            msg_dev.angle_y = float('nan')

        self.publisher.publish(msg_dev)

        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w//2, h//2), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if detect_result:
             cv2.putText(frame, f"Ang: X{self.angle_x:.1f} Y{self.angle_y:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)

    def destroy_node(self):
        self.stop_recording()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageDectPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()