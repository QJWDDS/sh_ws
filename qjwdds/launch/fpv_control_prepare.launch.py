#!/usr/bin python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    home_dir = os.path.expanduser('~')
    
    # 启动MicroXRCEAgent
    micro_xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
        name='micro_xrce_agent'
    )
    
    # 启动QGC
    qgc_process = ExecuteProcess(
        cmd=[os.path.join(home_dir, '下载/QGroundControl-x86_64.AppImage')],
        output='screen',
        name='qgroundcontrol'
    )
    
    # 启动PX4 Gazebo仿真
    px4_process = ExecuteProcess(
        cmd=['bash', '-c', f'cd {home_dir}/PX4-Autopilot && PX4_GZ_WORLD=default make px4_sitl gz_x500_mono_cam'],
        output='screen',
        name='px4_simulation'
    )

    return LaunchDescription([
        micro_xrce_agent,
        qgc_process,
        px4_process
    ])