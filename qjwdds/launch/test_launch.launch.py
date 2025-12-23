#!/usr/bin python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessStart

def generate_launch_description():

    # 生成球体
    spawn_sphere_cmd = f'gz service --service /world/default/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 1000 --req \'sdf_filename: "/home/shuai/sh_ws/src/qjwdds/gz/models/sptball.sdf"\''
    
    spawn_sphere_action = ExecuteProcess(
        cmd=['bash', '-c', spawn_sphere_cmd],
        output='screen',
        name='spawn_redball'
    )

    # 桥接
    gz_ros_bridge_cmd = f'ros2 run ros_gz_bridge parameter_bridge /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image'
    
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

    delay_rqt_event = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=gz_ros_bridge_action,
            on_start=[
                LogInfo(msg='已桥接...'),
                TimerAction(
                    period=3.0,  # 等待
                    actions=[rqt_action]
                )
            ]
        )
    )
    
    return LaunchDescription([
        spawn_sphere_action,
        gz_ros_bridge_action
    ])