# Overview

This repo packages practical, field-tested LiDAR–camera calibration scripts and utilities you’ve used in the lab. It focuses on:

Calibration: checkerboard or AprilTag board solves; export extrinsics as 4×4 T_lidar_cam and YAML/JSON.

Validation: reprojection error, point-to-pixel alignment overlays, and sanity-check visualizations.

Consumption: ready-to-use outputs for ROS TF, Open3D, and downstream perception.

✔️ The implementation code you provide will be kept exactly the same in logic; we only apply formatting/comments where helpful.

Features

Checkerboard or AprilTag target support.

Saves R and t and the homogeneous 4×4 transform.

Validates with reprojection metrics & pixel–point overlays.

ROS (bag/Topics)/offline file pipelines.

Config-driven (camera intrinsics, board specs, paths).

Supported Environments

OS: Ubuntu 20.04/22.04 (not tested but should work with some tweaks), macOS (limited), Windows WSL.

Python: 3.8–3.11

ROS: ROS1 Noetic (optional), ROS2 Humble (optional)

Key deps: OpenCV, NumPy, Open3D/PCL (optional), apriltag/pyapriltag (if used).
