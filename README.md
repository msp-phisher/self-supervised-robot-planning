# Self-Supervised Robot Planning (TAMP)

This project implements a Task and Motion Planning (TAMP) framework for a robotic manipulator (Franka Emika Panda) using a self-supervised perception system. The system learns to identify and cluster objects in a Gazebo simulation environment without manual labeling, then uses a PDDL-based planner to execute pick-and-place tasks.

## Demonstration

![Project Demo](media/demo.mp4)

## Project Overview

The system operates in three main phases:
1.  **Self-Supervised Learning:** Collecting random scene data and training a SimCLR (Contrastive Learning) model to extract robust visual features.
2.  **Unsupervised Grounding:** Using K-Means clustering to group visual features into symbolic identities (e.g., Red Box, Green Box, Blue Box).
3.  **Task & Motion Planning:** Generating PDDL problems from the perceived world state and executing motion plans via MoveIt 2.

## File Structure

*   `src/pytamp_moveit_bridge/`: Core logic for perception, planning, and execution.
*   `src/panda_moveit_config/`: Robot configuration and Gazebo world definitions.
*   `models/`: Saved neural network weights and clustering models.
*   `dataset/`: Images used for training the perception system.
*   `media/`: Project demonstration videos and screenshots.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/msp-phisher/self-supervised-robot-planning.git
cd self-supervised-robot-planning
```

### 2. Install Dependencies
Ensure you have ROS 2 (Humble or later) and the following Python packages installed:
```bash
pip install torch torchvision numpy opencv-python scikit-learn joblib pyperplan
```

### 3. Build the Workspace
```bash
colcon build
source install/setup.bash
```

## Step-by-Step Execution

### Step 1: Data Collection
Launch the simulation and run the data collector to generate a randomized dataset.
```bash
# In terminal 1: Launch Gazebo
ros2 launch panda_moveit_config moveit_gazebo_obb.py

# In terminal 2: Run data collector
python3 src/pytamp_moveit_bridge/scripts/collect_data.py
```
This saves 1,000 randomized images to the `dataset/images/` directory.

### Step 2: Training the Perception Model
Train the SimCLR backbone on the collected images.
```bash
python3 src/pytamp_moveit_bridge/scripts/train_simclr.py --epochs 100
```
This produces `models/simclr_backbone.pt`.

### Step 3: Clustering and Grounding
Run the feature extraction and K-Means clustering to create symbolic labels.
```bash
python3 src/pytamp_moveit_bridge/scripts/cluster_features.py
```
To visualize the clusters and verify the grouping:
```bash
python3 src/pytamp_moveit_bridge/scripts/visualize_clusters.py
```

### Step 4: Running the Planner
Start the full system and trigger a planning goal (e.g., picking the blue box).
```bash
# In terminal 1: Launch the full system
ros2 launch pytamp_moveit_bridge integration.launch.py

# In terminal 2: Send a goal to the planner
ros2 topic pub /planning_goal std_msgs/msg/String "{data: 'blue_box'}"
```

## Visualizing Results

The clustering process groups similar objects together. Below is an example of the features discovered by the self-supervised system:

![Cluster Samples](cluster_0_samples.png)

## License

This project is licensed under the MIT License.
