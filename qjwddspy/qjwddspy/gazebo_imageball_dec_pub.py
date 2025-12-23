#!/usr/bin python3
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
from cv_bridge import CvBridge

class ImageDectPublisher(Node):
    def __init__(self):
        super().__init__('gazebo_imageball_dec_pub')
        
        self.get_logger().set_level(logging.DEBUG)
        # 参数声明
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 30.0),
                ('calibration_file', '/home/shuai/sh_ws/src/qjwddspy/qjwddspy/usb_ca_0515.yaml'),
                ('video_save_dir', '/home/shuai/sh_ws/document/video'),
                ('recording_fps', 15.0),
                ('min_object_area', 30),#300
                ('hsv_lower', [0, 80, 100]),
                ('hsv_upper', [255, 255, 255]),
                ('circularity_threshold', 0.7),
                ('video_output', False),
                ('image_topic', '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image')
            ]
        )
        
        self.hsv_lower = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
        self.hsv_upper = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)
        self.min_area = self.get_parameter('min_object_area').value
        self.circularity_threshold = self.get_parameter('circularity_threshold').value
        self.video_output = self.get_parameter('video_output').value
        self.image_topic = self.get_parameter('image_topic').value
        
        if not (0 <= self.circularity_threshold <= 1):
            self.get_logger().warn(f"圆形度阈值{self.circularity_threshold}超出0-1范围，调整为0.7")
            self.circularity_threshold = 0.7
        
        self.video_save_dir = self.get_parameter('video_save_dir').value
        os.makedirs(self.video_save_dir, exist_ok=True)
        self.get_logger().info(f"视频保存到: {self.video_save_dir}")
        
        # 创建发布者和订阅者
        self.publisher = self.create_publisher(ImageDeviation, '/camera/image_deviation', 10)
        self.subscriber = self.create_subscription(
            Image, 
            self.image_topic, 
            self.image_callback, 
            10
        )
        
        # 图像转换
        self.bridge = CvBridge()
        
        # 相机标定参数
        self.K = None
        self.D = None
        self.load_calibration()
        
        # 视频录制相关
        self.video_writer = None
        self.is_recording = False
        if self.video_output:
            self.get_logger().info("视频录制已启用，将在收到第一帧图像后开始")
        
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
        self.current_circularity = 0.0
        self.last_frame = None

    def load_calibration(self):
        """加载相机标定参数"""
        calibration_file = self.get_parameter('calibration_file').value
        
        if not os.path.isfile(calibration_file):
            self.get_logger().error(f"标定文件不存在: {calibration_file}")
            return False
        
        try:
            fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                self.get_logger().error(f"无法打开标定文件: {calibration_file}")
                return False
                
            self.K = fs.getNode("K").mat()
            self.D = fs.getNode("D").mat()
            fs.release()
            
            if self.K is None or self.K.shape != (3, 3):
                self.get_logger().error(f"K矩阵无效: {self.K}")
                self.K = None
                self.D = None
                return False
                
            self.get_logger().info(f"成功加载标定参数")
            return True
        except Exception as e:
            self.get_logger().error(f"标定文件加载失败: {str(e)}")
            self.K = None
            self.D = None
            return False

    def start_recording(self, frame):
        """开始录制视频"""
        if self.is_recording:
            return
            
        height, width = frame.shape[:2]
        
        if height <= 0 or width <= 0:
            self.get_logger().error("无法获取有效图像尺寸，无法开始录制")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.video_save_dir, f"gz_ball_detection_{timestamp}.mkv")
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            video_path, 
            fourcc, 
            min(self.get_parameter('recording_fps').value, 30.0),
            (width, height)
        )
        
        if not self.video_writer.isOpened():
            self.get_logger().error(f"无法创建视频文件: {video_path}")
            self.video_writer = None
            return
            
        self.is_recording = True
        self.recording_start_time = time.time()
        self.get_logger().info(f"开始录制视频: {video_path}")

    def stop_recording(self):
        """停止录制视频"""
        if not self.is_recording:
            return
            
        if self.video_writer is not None:
            self.video_writer.release()
            duration = time.time() - self.recording_start_time
            self.get_logger().info(f"录制已停止，时长: {duration:.2f}秒")
            
        self.is_recording = False
        self.video_writer = None

    def detect_spherical_object(self, frame) -> Optional[Union[Tuple[int, int], Tuple[int, int, np.ndarray, float]]]:
        """检测特定颜色的最大球形物体，返回目标位置、轮廓和圆形度"""
        if frame is None or frame.size == 0:
            self.get_logger().debug("接收到空帧")
            return None
            
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
                
            best_contour = None
            best_circularity = 0.0
            
            # 筛选最佳球形轮廓
            for contour in contours:
                area = cv2.contourArea(contour)

                if area < self.min_area:
                    continue

                perimeter = cv2.arcLength(contour, closed=True)
                if perimeter < 1e-6:  # 避免除零错误
                    continue
                
                # 计算圆形度: 4π*面积/(周长²)
                circularity = 4 * np.pi * area / (perimeter **2)

                if circularity > self.circularity_threshold and circularity > best_circularity:
                    best_circularity = circularity
                    best_contour = contour

            if best_contour is None:
                self.get_logger().debug(f"未找到符合圆形度阈值({self.circularity_threshold})的物体")
                return None
                
            # 计算目标中心
            M = cv2.moments(best_contour)
            if M["m00"] < 1:
                return None
                
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            return (cX, cY, best_contour, best_circularity)
            
        except Exception as e:
            self.get_logger().error(f"球形物体检测错误: {str(e)}")
            return None

    def calculate_angles(self, err_u, err_v) -> Tuple[float, float]:
        """根据图像偏差计算空间角度"""
        if self.K is None:
            return float('nan'), float('nan')
            
        try:
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            
            if abs(fx) < 0.1 or abs(fy) < 0.1:  
                return float('nan'), float('nan')
                
            angle_x = np.degrees(np.arctan(err_u / fx))
            angle_y = np.degrees(np.arctan(err_v / fy))
            return angle_x, angle_y
            
        except Exception as e:
            self.get_logger().error(f"角度计算错误: {str(e)}")
            return float('nan'), float('nan')
    
    def add_debug_info(self, frame, detected: bool, contour=None, circularity=0.0):
        """在视频帧上添加调试信息"""
        if frame is None:
            return frame
            
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2

        cv2.drawMarker(frame, (center_x, center_y), (255, 0, 0), 
                      markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
        
        if detected and contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            if self.target_position:
                cv2.circle(frame, self.target_position, 5, (0, 0, 255), -1)
                cv2.line(frame, (center_x, center_y), self.target_position, (255, 0, 255), 2)
            
            # 显示信息
            cv2.putText(frame, f"Error U: {self.err_u:.1f}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Error V: {self.err_v:.1f}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Angle X: {self.angle_x:.2f}deg", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Angle Y: {self.angle_y:.2f}deg", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Circularity: {circularity:.2f}", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (width - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Circ Thresh: {self.circularity_threshold:.2f}", (width - 250, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame

    def image_callback(self, msg):
        """处理接收到的图像消息"""
        self.frame_count += 1
        # 计算FPS（2秒更新一次）
        current_time = time.time()
        if current_time - self.last_fps_time > 2.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        try:
            # 将ROS图像消息转换为OpenCV格式bgr8
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_frame = frame
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {str(e)}")
            return
        
        if self.video_output and not self.is_recording:
            self.start_recording(frame)
        
        # 检测球形物体
        result = self.detect_spherical_object(frame)
        detected = result is not None
        self.current_circularity = 0.0
        
        if detected:
            cX, cY, contour, circularity = result
            self.target_position = (cX, cY)
            self.current_circularity = circularity
            
            # 计算偏差
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            self.err_u = cX - center_x
            self.err_v = cY - center_y
            
            # 计算角度
            self.angle_x, self.angle_y = self.calculate_angles(self.err_u, self.err_v)
            
            # 发布消息
            msg = ImageDeviation()
            msg.err_u = float(self.err_u)
            msg.err_v = float(self.err_v)
            msg.angle_x = float(self.angle_x)
            msg.angle_y = float(self.angle_y)
            self.publisher.publish(msg)
            
            # 日志信息
            self.get_logger().info(
                f"目标位置: ({cX},{cY}) | 偏差: u={self.err_u:.1f}, v={self.err_v:.1f} | "
                f"角度: x={self.angle_x:.2f}°, y={self.angle_y:.2f}° | 圆形度: {circularity:.2f}",
                throttle_duration_sec=0.5)
            
            self.last_detection_time = current_time
        else:
            if current_time - self.last_detection_time > 3.0:
                self.get_logger().warning("未检测到符合条件的球形物体", throttle_duration_sec=1)
                
            msg = ImageDeviation()
            msg.err_u = float('nan')
            msg.err_v = float('nan')
            msg.angle_x = float('nan')
            msg.angle_y = float('nan')
            self.publisher.publish(msg)
        
        debug_frame = frame.copy()
        debug_frame = self.add_debug_info(
            debug_frame, 
            detected, 
            contour if detected else None,
            self.current_circularity if detected else 0.0
        )
        
        # 保存视频
        if self.is_recording and self.video_writer is not None:
            try:
                self.video_writer.write(debug_frame)
            except Exception as e:
                self.get_logger().error(f"写入视频失败: {str(e)}")
                self.stop_recording()

    def destroy_node(self):
        """销毁节点并释放资源"""
        self.get_logger().info("开始清理资源...")
        
        if self.is_recording:
            self.stop_recording()

        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageDectPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点被手动中断")
    except Exception as e:
        node.get_logger().fatal(f"未处理的异常: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()