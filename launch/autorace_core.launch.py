from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_name = 'autorace_core_GBTBoys'

    # Get package share directory
    pkg_share = FindPackageShare(pkg_name).find(pkg_name)

    return LaunchDescription([
        Node(
            package=pkg_name,
            executable='competition',
            name='competition',
            output='screen'
        )
    ])