#!/usr/bin python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():

    # 桥接
    gz_ros_bridge_cmd = f'ros2 run ros_gz_bridge parameter_bridge /world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image'
    
    gz_ros_bridge_action = ExecuteProcess(
        cmd=['bash', '-c', gz_ros_bridge_cmd],
        output='screen',
        name='gz_ros_bridge'
    )
    
    #rqt
    rqt_action = ExecuteProcess(
        cmd=['ros2', 'run', 'rqt_image_view', 'rqt_image_view'],
        output='screen',
        name='rqt_action'
    )

    gazebo_node = Node(
        package='qjwddspy',
        executable='gazebo_imageball_dec_pub',
        name='gazebo_pub',
        output='screen',
        parameters=[{
            'hsv_lower': [0, 80, 100],  
            'hsv_upper': [255, 255, 255],
            'circularity_threshold': 0.7,
            'video_output': False,
            'min_object_area':100,
            'image_topic':'/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image'
        }]
    )

    return LaunchDescription([
        gz_ros_bridge_action,
        rqt_action,
        gazebo_node
    ])