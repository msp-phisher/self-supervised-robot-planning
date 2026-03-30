# Self-Supervised Robot Planning (TAMP)

This project implements a Task and Motion Planning (TAMP) framework for a robotic manipulator (Franka Emika Panda) using a self-supervised perception system. The system learns to identify and cluster objects in a Gazebo simulation environment without manual labeling.

## Demonstration

Visit the link below to view the demonstration video:
[Project Demo Video](media/demo.mp4)

## Project Overview

The system operates in three main phases:
1.  **Self-Supervised Learning:** Collecting random scene data and training a SimCLR (Contrastive Learning) model to extract robust visual features.
2.  **Unsupervised Grounding:** Using K-Means clustering to group visual features into symbolic identities (e.g., Red Box, Green Box, Blue Box).
3.  **Task & Motion Planning:** Generating PDDL problems from the perceived world state and executing motion plans via MoveIt 2.

## File Structure

*   `src/`: Contains the ROS 2 packages for perception, planning, and execution.
*   `models/`: Stores the trained neural networks and clustering models.
*   `dataset/`: Images used for training the perception system.
*   `media/`: Project demonstration videos and screenshots.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/msp-phisher/self-supervised-robot-planning.git
cd self-supervised-robot-planning
```

### 2. Install Dependencies
```bash
pip install torch torchvision numpy opencv-python scikit-learn joblib pyperplan
```

### 3. Build the Workspace
```bash
colcon build
source install/setup.bash
```

## Running the System (Four Terminal Setup)

To run the full TAMP system with the Franka Panda robot, follow these steps across four separate terminals.

### Terminal 1: Simulation and MoveIt
Launch the Gazebo simulator and the MoveIt motion planning environment.
```bash
source install/setup.bash
ros2 launch panda_moveit_config moveit_gazebo_obb.py
```

### Terminal 2: Perception
Run the perception node that processes camera images and identifies objects using the self-supervised models.
```bash
source install/setup.bash
ros2 run pytamp_moveit_bridge ssl_perception_node.py
```

### Terminal 3: Planner
Start the PDDL planner node that generates task plans based on the live world state.
```bash
source install/setup.bash
ros2 run pytamp_moveit_bridge planner_node.py
```

### Terminal 4: Action Executor and Goal Trigger
Start the action executor and then publish a goal to start the picking task.
```bash
# Start the executor
source install/setup.bash
ros2 run pytamp_moveit_bridge action_executor.py

# In the same terminal (or another), trigger the goal:
ros2 topic pub /planning_goal std_msgs/msg/String "{data: 'blue_box'}"
```

## Training Lifecycle

To re-train or update the models, follow these steps:

1.  **Data Collection:** `python3 src/pytamp_moveit_bridge/scripts/collect_data.py`
2.  **Training SimCLR:** `python3 src/pytamp_moveit_bridge/scripts/train_simclr.py`
3.  **Clustering:** `python3 src/pytamp_moveit_bridge/scripts/cluster_features.py`
4.  **Visualization:** `python3 src/pytamp_moveit_bridge/scripts/visualize_clusters.py`

## License

This project is licensed under the MIT License.
