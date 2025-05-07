import os
import launch

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory

def generate_launch_description():
    data_path_arg = DeclareLaunchArgument('data_path', default_value='/userdata/dev_front/parse_res.txt')
    model_path_arg = DeclareLaunchArgument('model_path', default_value='/userdata/dev_ws/src/originbot/originbot_deeplearning/line_follower_perception/model/resnet18_224x224_nv12_maze.bin')
    go_time = DeclareLaunchArgument('go_time', default_value='1.0')
    turning_vth = DeclareLaunchArgument('turning_vth', default_value='2.0')
    straight_vth = DeclareLaunchArgument('straight_vth', default_value='1.5')
    straight_turning_vth = DeclareLaunchArgument('straight_turning_vth', default_value='0.08')

    line_follower_perception_node = Node(
        package='line_follower_perception',
        executable='line_follower_maze', 
        output='screen',
        emulate_tty=True,
        parameters=[{
                'data_path': LaunchConfiguration('data_path'), 
                'model_path': LaunchConfiguration('model_path'),
                'go_time': LaunchConfiguration('go_time'),
                'turning_vth': LaunchConfiguration('turning_vth'),
                'straight_vth': LaunchConfiguration('straight_vth'),
                'straight_turning_vth': LaunchConfiguration('straight_turning_vth')
        }]
    )

    robot_base_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('originbot_bringup'),
                'launch/originbot.launch.py'))
    )

    usb_camera_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('line_follower_perception'),
                'launch/usb_cam_web.launch.py'))
    )

    return LaunchDescription([
        data_path_arg,
        model_path_arg,
        go_time,
        turning_vth,
        straight_vth,
        straight_turning_vth,
        line_follower_perception_node,
        usb_camera_node,
        robot_base_node
    ])
