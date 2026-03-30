from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    
    # 1. Build MoveIt Config
    # Uses panda.urdf.xacro with ros2_control_hardware_type:=mock_components
    moveit_config = (
        MoveItConfigsBuilder(robot_name="panda", package_name="panda_moveit_config")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": "mock_components/GenericSystem", "initial_positions_file": "initial_positions.yaml"}
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .moveit_cpp(file_path="config/controller_setting.yaml") # Relative path in package
        .to_moveit_configs()
    )

    # 2. Nodes
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control_node (Mock Hardware)
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            os.path.join(get_package_share_directory("panda_moveit_config"), "config", "ros2_controllers.yaml"),
        ],
        output="both",
    )

    # Move Group
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"publish_robot_description_semantic": True},
            {"use_sim_time": False}, 
        ],
    )

    # RViz
    rviz_config_file = os.path.join(
        get_package_share_directory("panda_moveit_config"),
        "config",
        "motion_planning.rviz",
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
    )

    # Spawners
    load_controllers = []
    for controller in ["panda_arm_controller", "panda_hand_controller", "joint_state_broadcaster"]:
        load_controllers += [
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller, "--controller-manager", "/controller_manager"],
                output="screen",
            )
        ]

    # 3. TAMP Nodes (Included directly for convenience)
    
    # Mock Perception
    mock_perception = Node(
        package='pytamp_moveit_bridge',
        executable='mock_perception.py',
        name='mock_perception',
        output='screen'
    )

    # Object State Manager
    state_manager = Node(
        package='pytamp_moveit_bridge',
        executable='object_state_manager.py',
        name='object_state_manager',
        output='screen'
    )
    
    # Planner
    planner = Node(
        package='pytamp_moveit_bridge',
        executable='planner_node.py',
        name='mcts_planner',
        output='screen'
    )
    
    # Executor
    # Note: action_executor.py handles MoveItPy internally using MoveItConfigsBuilder.
    # It just needs to run. 
    executor = Node(
        package='pytamp_moveit_bridge',
        executable='action_executor.py',
        name='action_executor',
        output='screen'
    )
    
    # Static TF for camera (Mock Perception frame)
    # mock_perception uses "camera_link". robot_state_publisher publishes "camera_link" from URDF.
    # BUT robot_state_publisher needs joint values.
    # mock_components/GenericSystem provides joint states (default 0).
    # So panda_link0 -> camera_link should be valid via URDF static joint.
    # Wait, panda.urdf.xacro has <xacro:camera_v0 parent="panda_link0">
    # That creates a fixed joint. RSP publishes fixed joints.
    # So TF should work automatically without static_transform_publisher!
    
    nodes_to_start = [
        robot_state_publisher,
        ros2_control_node,
        run_move_group_node,
        rviz_node,
        mock_perception,
        state_manager,
        planner,
        executor
    ] + load_controllers

    return LaunchDescription(nodes_to_start)
