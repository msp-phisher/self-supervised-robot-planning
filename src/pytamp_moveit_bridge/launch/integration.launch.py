from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    
    # 1. Include the Gazebo + MoveIt Launch (The Foundation)
    # This comes from panda_moveit_config/launch/moveit_gazebo_obb.py
    panda_config_dir = get_package_share_directory('panda_moveit_config')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(panda_config_dir, 'launch', 'moveit_gazebo_obb.py')
        )
    )

    # 2. YOLOv8 OBB Publisher (Perception)
    yolov8_node = Node(
        package='yolov8_obb',
        executable='yolov8_obb_publisher.py',
        name='yolov8_obb_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 3. Object State Manager (Bridge Perception -> World)
    state_manager = Node(
        package='pytamp_moveit_bridge',
        executable='object_state_manager.py',
        name='object_state_manager',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # 4. Planner Node (The Brain)
    planner = Node(
        package='pytamp_moveit_bridge',
        executable='planner_node.py',
        name='mcts_planner',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # 5. Action Executor (The Muscle)
    executor = Node(
        package='pytamp_moveit_bridge',
        executable='action_executor.py',
        name='action_executor',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 6. RQT Image View (Optional, for user debug)
    image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='camera_viewer',
        arguments=['/camera/image_raw'], 
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    return LaunchDescription([
        gazebo_launch,
        yolov8_node,
        state_manager,
        planner,
        executor,
        image_view
    ])
