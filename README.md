# realsense_pointnet_on_ROS


# PointNet with ROS Integration

This project demonstrates the integration of PointNet with ROS to process point clouds obtained from a Realsense D455 camera. The PointNet model is used to segment potential movable objects such as humans, robots, and chairs.

## Features
- Implementation of PointNet using PyTorch
- Integration with ROS
- Processing of point clouds from Realsense D455 camera

## Requirements
- Python 3.7 or later
- PyTorch
- ROS Noetic
- Realsense ROS package
- OpenCV
- NumPy

## Code Structure
- `src/`
  - `pointnet.py`: Contains the implementation of the PointNet model.
  - `ros_pointnet.py`: Contains the ROS node for integrating PointNet.
  - `mvs.py`: Contains the MVS algorithm implementation for generating point clouds.
- `launch/`
  - `realsense_pointnet.launch`: Launch file for starting the ROS nodes.
- `scripts/`
  - `mvs_node.py`: ROS node for running the MVS algorithm.
  - `pointnet_node.py`: ROS node for running PointNet.



## Setup

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ros_pointnet.git
   cd ros_pointnet

3.  Build the ROS package:
     ```bash
     catkin_make
     source devel/setup.bash

## Usage
To launch the ROS nodes:
   ```bash
   roslaunch ros_pointnet realsense_pointnet.launch




