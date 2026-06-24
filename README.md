# H2R Handovers

# Repo structure

h2r_handoves
- launch
    - realsense launch
    - 

- nodes
    - perception: subscribes to camera topic, runs egohos, if object-in-hand, filters both rgb and pcl, publishes them on other topic
    - grasp estimation: subscribes to rgb & d filtred topics, runs graspnett, publishes grasp poses
    - handover scenario: subscribes to grasp pose , chooses best one, and publishes to moveit servo. w

handover_ws/                       # build here:  colcon build --symlink-install
└── src/
    ├── EgoHOS/                    # your fork — pip -e into egohos_venv;  add COLCON_IGNORE
    ├── graspnet-baseline/         # your fork — pip -e into graspnet_venv; add COLCON_IGNORE
    ├── franka_ros2/               # external — drivers, controllers, moveit config
    ├── moveit_servo/              # external (or apt: ros-humble-moveit-servo → omit from src)
    └── handover/                  # your NEW repo (one git repo, two packages)
        ├── handover_interfaces/   # ament_cmake — custom msgs
        │   ├── msg/
        │   │   ├── GraspCandidate.msg
        │   │   ├── GraspCandidates.msg
        │   │   └── SelectedGrasp.msg
        │   ├── CMakeLists.txt
        │   └── package.xml
        └── handover/              # ament_python — nodes, policies, launch, config
            ├── handover/
            │   ├── egohos_node.py
            │   ├── graspnet_node.py
            │   ├── grasp_selection_node.py
            │   ├── handover_orchestrator.py
            │   └── policies/      # depth_heuristic.py, comfort_aware.py (your ablation seam)
            ├── launch/handover.launch.py
            ├── config/servo_params.yaml
            ├── setup.py
            └── package.xml

# Implementation steps

2. install & verify graspnet
3. install & verify realsense

4. integrate realsense & egohos and their launch and interfaces
5. add in graspnet
6. add in dummy selection and moveit planning and execution
 9. finalize this pipeline launch docs configs

10. Add in intention analysis, aka hand tracking with egohos that publishes only when hand has stabilized
10. add in proper selection node policy  
11. add in proper grasping which doesnt move away unless there the hand let the weight go with fts
12. run whole pipeline a lot
13. design userstudy usecase with and without the selection criteria and the grasp sensing
14. do user study

# Installation

## Workspace setup

- ubuntu 22 + rt kernel
- ros2 humble
- cuda 13
- Franka Version 4.2.2
- libfranka (By LCAS) version 0.9.2
- Clone .repos

## Install EgoHOS

1. Create the venv with the same Python as your ROS distro (Humble = 3.10) so rclpy will import later:

python3 -m venv --system-site-packages egohos_venv
source egohos_venv/bin/activate

2. Your EgoHOS install, now running inside the venv:

cd ./src/EgoHOS
pip install -r requirements.txt

3. verify torch + cuda:
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
python -c "import torch; x=torch.rand(3).cuda(); print(x+1)"

Note: If it doesn't work: pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

4. install egohos dependencies
pip install -U openmim
if it doesn't work: 
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
python -c "import mmcv; from mmcv.ops import nms; print(mmcv.__version__)"

mim install mmcv-full==1.6.0
cd mmsegmentation
pip install -v -e .

5. Download required resources

bash download_datasets.sh
bash download_checkpoints.sh
bash download_testimages.sh

## Install graspnet

/home/franka/projects/handovers_ws/src/graspnet_venv/bin/pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130

# Quick Tests

## Robot Movement

ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2

## EgoHOS Segmentation

cd ./src/EgoHOS/mmsegmentation
bash pred_all_obj1.sh

# Handover Demo