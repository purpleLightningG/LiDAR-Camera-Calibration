# Overview

This repo packages practical, field-tested LiDAR–camera calibration scripts and utilities developed in the lab. It focuses on:

- **Calibration**: Checkerboard or AprilTag board solves; exports extrinsics as 4×4 `T_lidar_cam` and YAML/JSON.
- **Validation**: Reprojection error, point-to-pixel alignment overlays, and sanity-check visualizations.
- **Consumption**: Ready-to-use outputs for ROS TF, Open3D, and downstream perception.

---

## Features
- Checkerboard **or** AprilTag target support.  
- Saves **R** and **t** along with the homogeneous **4×4 transform**.  
- Validates with reprojection metrics & pixel–point overlays.  
- ROS (bag/Topics) and offline file pipelines.  
- Config-driven (camera intrinsics, board specs, paths).  

---

## Supported Environments
- **OS**: Ubuntu 20.04 / 22.04 (not tested but should work with minor tweaks), macOS (limited), Windows WSL.  
- **Python**: 3.8 – 3.11  
- **ROS**: ROS1 Noetic (optional), ROS2 Humble (optional)  
- **Key Dependencies**: OpenCV, NumPy, Open3D / PCL (optional), apriltag / pyapriltag (if used).  

---

## Quick Start
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

## Set your private dataset paths inside the scripts (RGB/PCD/DATA_PATH)
python scripts/smallCubeCal.py
