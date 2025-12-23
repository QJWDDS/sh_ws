#!/usr/bin python3
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
   
    # 自定义节点
    gazebo_node = Node(
        package='qjwddspy',
        executable='gazebo_imageball_dec_pub',
        name='gazebo_pub',
        output='screen',
        parameters=[{
            'hsv_lower': [0, 80, 100],  
            'hsv_upper': [255, 255, 255],
            'circularity_threshold': 0.7,
            'video_output': True
        }]
    )
  
    control_node = Node(
        package='qjwdds',
        executable='sh_vision_control_s',
        name='flight_controller',
        output='screen',
        parameters=[{
            'takeoff_relative_altitude': 5.0,  
            'proportional_gain': 5.0,#2
            'vertical_gain': 2.0,#1.5
            'yaw_gain': 1.0,#0.5
            'max_speed': 8.0,#
            'max_vertical_speed': 2.0,#1.0
            'max_yaw_rate': 1.0,#0.5
            'max_relative_altitude': 10.0,
            'target_loss_timeout': 0.5,
            'target_loss_max_count':25
        }]
    )
 
    return LaunchDescription([
        TimerAction(period=2.0, actions=[gazebo_node]),
        TimerAction(period=5.0, actions=[control_node])
    ])
