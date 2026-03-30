import os
from sys import prefix
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return file.read()
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def generate_launch_description():

    # Command-line arguments
    tutorial_arg = DeclareLaunchArgument(
        "rviz_tutorial", default_value="False", description="Tutorial flag"
    )

    kinematics_yaml = load_yaml(
        "panda_moveit_config", "config/kinematics.yaml"
    )

    # Load Robot Description
    robot_description_config = Command(
        [FindExecutable(name="xacro"), " ", os.path.join(get_package_share_directory("panda_moveit_config"), "config", "panda.urdf.xacro")]
    )
    robot_description = {"robot_description": robot_description_config}

    # Load Robot Description Semantic
    with open(os.path.join(get_package_share_directory("panda_moveit_config"), "config", "panda.srdf"), "r") as file:
        robot_description_semantic_config = file.read()
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_config}

    # RViz
    tutorial_mode = LaunchConfiguration("rviz_tutorial")
    rviz_base = os.path.join(get_package_share_directory("panda_moveit_config"), "launch")
    rviz_full_config = os.path.join(rviz_base, "moveit.rviz")
    rviz_empty_config = os.path.join(rviz_base, "moveit_empty.rviz")
    rviz_node_tutorial = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_empty_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
        ],
        condition=IfCondition(tutorial_mode),
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_full_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
        ],
        condition=UnlessCondition(tutorial_mode),
    )

    return LaunchDescription(
        [
            tutorial_arg,
            rviz_node,
            rviz_node_tutorial,
        ]
    )
