# H2R Handovers

Human-to-Robot handover pipeline for the Franka Emika Panda, built on ROS 2 Humble.

A human holds out an object → the system segments the hand and object ([EgoHOS](https://github.com/YaraShahin/EgoHOS)), generates grasp candidates ([GraspNet](https://github.com/YaraShahin/graspnet-baseline)), selects the best grasp, and commands the robot to take the object.

## Architecture

```
RealSense D435 (640×480 @ 30fps)
       │
       ├─ /camera/.../color/image_raw ──────► EgoHOS Node (egohos_venv)
       │                                         │
       │                                         ├─ segmentation_mask ──► Hand Stabilization Node
       │                                         │                              │
       │                                         │                         hand_stability_status
       │                                         │                              │
       ├─ /camera/.../aligned_depth_to_color ────┼──────────────────────► GraspNet Node (graspnet_venv)
       │                                         │                              │
       └─ /camera/.../camera_info ───────────────┘                        grasp_candidates (PoseArray)
                                                                                │
                                                                         [Grasp Selection — TBD]
                                                                                │
                                                                         [Handover Orchestrator — TBD]
                                                                                │
                                                                          MoveIt / Franka ROS2
```

### Nodes

| Node | Package / Location | What it does |
|------|--------------------|-------------|
| **egohos_node** | `EgoHOS/scripts/driver.py` (own venv) | Subscribes to RGB, runs 3-stage segmentation cascade (hands → contact boundary → object), publishes `segmentation_mask` (mono8: 0=bg, 1=hand, 2=object) and optional colour overlay |
| **hand_stabilization_node** | `h2r_handovers` | Tracks hand centroid in the segmentation mask; publishes edge-triggered status (`hand_stable` / `hand_unstable` / `no_hand`). Note: Only used for initial grasp triggering, not during approach since hand cannot be detected well during movement due to camera angle change. |
| **graspnet_node** | `graspnet-baseline/scripts/driver.py` (own venv) | Waits for `hand_stable`, time-syncs mask + depth + CameraInfo, builds object point cloud, runs GraspNet, publishes scored `PoseArray` + debug image |
| **grasp_selection_node** | *TBD* | Choose best grasp from candidates (depth heuristic, comfort-aware, etc.) |
| **handover_orchestrator** | *TBD* | State machine: wait → approach → grip → detect release (F/T) → retract |

### Topics

| Topic | Type | Publisher | Subscriber(s) |
|-------|------|-----------|---------------|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | RealSense | egohos_node, graspnet_node (debug) |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | RealSense | graspnet_node |
| `/camera/camera/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | RealSense | graspnet_node |
| `segmentation_mask` | `sensor_msgs/Image` (mono8) | egohos_node | hand_stabilization_node, graspnet_node |
| `segmentation_overlay` | `sensor_msgs/Image` (bgr8) | egohos_node | RViz2 |
| `hand_stability_status` | `std_msgs/String` | hand_stabilization_node | graspnet_node |
| `grasp_candidates` | `geometry_msgs/PoseArray` | graspnet_node | *(selection node — TBD)* |
| `grasp_debug_image` | `sensor_msgs/Image` (bgr8) | graspnet_node | RViz2 |

## Workspace Setup

### Prerequisites

- Ubuntu 22.04 + RT kernel
- ROS 2 Humble
- CUDA ≥ 11.3
- Franka FCI firmware 4.2.2 / libfranka 0.9.2 (LCAS fork)

### Clone the workspace

```bash
mkdir -p ~/handover_ws/src && cd ~/handover_ws/src

# This repo
git clone https://github.com/<your-org>/h2r_handovers.git

# External deps (franka_ros2, EgoHOS, graspnet-baseline)
vcs import < h2r_handovers/humble.repos
```

The resulting layout:

```
handover_ws/
└── src/
    ├── EgoHOS/                 # segmentation — pip -e into egohos_venv; add COLCON_IGNORE
    ├── graspnet-baseline/      # grasping    — pip -e into graspnet_venv; add COLCON_IGNORE
    ├── franka_ros2/            # robot drivers + MoveIt config
    └── h2r_handovers/          # this repo — handover nodes, launch, config
```

### Install EgoHOS

1. Create a venv with system-site-packages (so `rclpy` is importable):

```bash
cd ~/handover_ws/src
python3 -m venv --system-site-packages egohos_venv
source egohos_venv/bin/activate
```

2. Install EgoHOS dependencies:

```bash
cd EgoHOS
pip install -r requirements.txt
```

3. Verify PyTorch + CUDA:

```bash
python -c "import torch; print(torch.__version__); x=torch.rand(3).cuda(); print(x+1)"
```

> If CUDA doesn't work, install a matching torch build:
> ```bash
> pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
>     --extra-index-url https://download.pytorch.org/whl/cu113
> ```

4. Install MMSegmentation:

```bash
pip install -U openmim
mim install mmcv-full==1.6.0
cd mmsegmentation
pip install -v -e .
```

> If `mim install` fails, try the direct wheel:
> ```bash
> pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
> ```

5. Verify mmcv:

```bash
python -c "import mmcv; from mmcv.ops import nms; print(mmcv.__version__)"
```

6. Download model weights and test data:

```bash
cd ~/handover_ws/src/EgoHOS
bash download_checkpoints.sh
bash download_datasets.sh    # optional, for offline testing
bash download_testimages.sh  # optional, for offline testing
```

7. Mark as COLCON_IGNORE:

```bash
touch ~/handover_ws/src/EgoHOS/COLCON_IGNORE
```

### Install GraspNet

1. Create a separate venv:

```bash
cd ~/handover_ws/src
python3 -m venv --system-site-packages graspnet_venv
source graspnet_venv/bin/activate
```

2. Install dependencies:

```bash
cd graspnet-baseline
pip install -r requirements.txt

# Install custom CUDA ops
cd pointnet2 && python setup.py install && cd ..
cd knn && python setup.py install && cd ..

# Install graspnetAPI
pip install graspnetAPI
```

3. Install PyTorch matching your CUDA:

```bash
pip install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118  # adjust for your CUDA
```

4. Download pretrained checkpoint (`checkpoint-rs.tar` — RealSense model):

```
Place at: graspnet-baseline/logs/log_rs/checkpoint.tar
```

5. Mark as COLCON_IGNORE:

```bash
touch ~/handover_ws/src/graspnet-baseline/COLCON_IGNORE
```

### Build the ROS 2 workspace

```bash
cd ~/handover_ws
colcon build --symlink-install
source install/setup.bash
```

## Quick Start

### 1. Verify robot movement (no vision)

```bash
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2
```

### 2. Verify EgoHOS segmentation (offline)

```bash
source ~/handover_ws/src/egohos_venv/bin/activate
cd ~/handover_ws/src/EgoHOS/mmsegmentation
bash pred_all_obj1.sh
# Check output in ../testimages/pred_obj1_vis/
```

### 3. Launch the handover pipeline

Terminal 1 — Camera + hand stabilization + RViz:
```bash
source ~/handover_ws/install/setup.bash
ros2 launch h2r_handovers handover.launch.xml
```

Terminal 2 — EgoHOS node (in its venv):
```bash
source ~/handover_ws/src/egohos_venv/bin/activate
python ~/handover_ws/src/EgoHOS/scripts/driver.py
```

Terminal 3 — GraspNet node (in its venv):
```bash
source ~/handover_ws/src/graspnet_venv/bin/activate
python ~/handover_ws/src/graspnet-baseline/scripts/driver.py
```

### 4. Debug / Visualize

In RViz2, add Image displays for:
- `segmentation_overlay` — see EgoHOS output
- `grasp_debug_image` — see GraspNet grasp candidates projected onto the image

Monitor hand stability:
```bash
ros2 topic echo hand_stability_status
```

Monitor grasp candidates:
```bash
ros2 topic echo grasp_candidates
```

## Implementation Roadmap

- [x] Install & verify EgoHOS
- [x] Install & verify GraspNet
- [x] Install & verify RealSense
- [x] Integrate RealSense + EgoHOS as ROS 2 nodes with launch and topics
- [x] Add GraspNet as ROS 2 node
- [x] Add hand stabilization (intention detection via centroid tracking)
- [x] Add grasp selection node with pluggable policies
- [x] Add handover orchestrator with MoveIt planning & execution
- [x] Finalize full pipeline launch, docs, and configs
- [ ] Add proper grasping with F/T release detection
- [ ] End-to-end testing
- [ ] Design user study (with/without selection criteria and grasp sensing)
- [ ] Run user study

---

## Archived Notes

> The following are original working notes kept for reference.

### Original repo structure plan

```
handover_ws/
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
```

### Original implementation step notes

```
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
```

### GraspNet venv torch install command

```bash
/home/franka/projects/handovers_ws/src/graspnet_venv/bin/pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130
```