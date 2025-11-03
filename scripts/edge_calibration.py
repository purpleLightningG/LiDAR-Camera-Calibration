#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import open3d as o3d
from pyapriltags import Detector
from sklearn.cluster import DBSCAN

# --- USER SETTINGS ---
RGB_FOLDER = r"folder/to/rgb"
PCD_FOLDER = r"folder/to/pcd"
SAVE_MATRIX = "extrinsic_calibration_cube.txt"
TAG_ID = 8
TAG_SIZE = 0.10  # meters (10 cm)
FX, FY = 912.0468, 911.8128
CX, CY = 962.5100, 546.4423
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=float)

# --- ALGORITHM PARAMETERS ---
RANSAC_DIST = 0.01
RANSAC_ITERS = 1000
MIN_PLANE_INLIERS = 40

CUBE_SIDE_M = 0.11  # meters (11 cm)
CUBE_TOL_M = 0.02  # ±2 cm tolerance

# --- HELPER FUNCTIONS ---


def list_paired_frames(rgb_folder, pcd_folder):
    """Finds all matching RGB and PCD files."""
    rgbs = sorted(
        [f for f in os.listdir(rgb_folder) if f.lower().endswith((".png", ".jpg"))]
    )
    pcds = sorted([f for f in os.listdir(pcd_folder) if f.lower().endswith(".pcd")])
    common_names = sorted(
        list(
            set(os.path.splitext(f)[0] for f in rgbs)
            & set(os.path.splitext(f)[0] for f in pcds)
        )
    )
    pairs = []
    for name in common_names:
        for ext in (".png", ".jpg"):
            rgb_path = os.path.join(rgb_folder, name + ext)
            if os.path.exists(rgb_path):
                pairs.append((rgb_path, os.path.join(pcd_folder, name + ".pcd"), name))
                break
    return pairs


def detect_apriltag_pose(gray):
    """Detects the target AprilTag and returns its plane equation."""
    det = Detector(families="tag36h11")
    tags = det.detect(
        gray, estimate_tag_pose=True, camera_params=[FX, FY, CX, CY], tag_size=TAG_SIZE
    )
    for t in tags:
        if t.tag_id == TAG_ID:
            R = t.pose_R
            tvec = t.pose_t.flatten()
            normal_vector = R[:, 2]
            d = -np.dot(normal_vector, tvec)
            return np.append(normal_vector, d)
    return None


# --- THIS IS THE NEW INTERACTIVE FUNCTION ---
def find_planes_in_manual_region(pcd):
    """
    Allows the user to manually select a region in the point cloud,
    then runs RANSAC to find cube planes within that region.
    """
    print("\n" + "=" * 50)
    print("ACTION: Please select a region containing the robot base.")
    print(" 1. A 3D view will open. Rotate to a top-down or side view.")
    print(" 2. Hold [Shift] and Left-Click to draw a polygon around the robot base.")
    print(" 3. Press [Enter] to confirm your selection.")
    print("=" * 50)

    # Use Open3D's interactive polygon selection
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Select Region of Interest")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    # Get the indices of the points inside the user-drawn polygon
    selected_indices = vis.get_picked_points()
    if len(selected_indices) < 50:
        print("  > Not enough points selected. Skipping.")
        return []

    # Create a new point cloud containing only the selected points
    pcd_roi = pcd.select_by_index(selected_indices)
    points_in_roi = np.asarray(pcd_roi.points)

    print(f"  > Selected {len(points_in_roi)} points. Searching for planes...")

    candidate_planes = []
    remaining_pts = points_in_roi.copy()

    # Sequentially find up to 3 planes within the selected region
    for _ in range(3):
        if len(remaining_pts) < MIN_PLANE_INLIERS:
            break

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(remaining_pts)

        plane_model, inlier_indices = temp_pcd.segment_plane(
            RANSAC_DIST, 3, RANSAC_ITERS
        )

        if len(inlier_indices) < MIN_PLANE_INLIERS:
            break

        plane_inliers = remaining_pts[inlier_indices]

        # Validate the size of the found plane
        try:
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(plane_inliers)
            )
            extents = sorted(obb.extent)

            if (
                abs(extents[1] - CUBE_SIDE_M) < CUBE_TOL_M
                and abs(extents[2] - CUBE_SIDE_M) < CUBE_TOL_M
            ):
                candidate_planes.append((plane_model, plane_inliers))
        except RuntimeError:
            continue

        # Remove the found plane and search for more
        remaining_pts = np.delete(remaining_pts, inlier_indices, axis=0)

    return candidate_planes


def visualize_candidate_planes(full_pcd, candidates):
    """Shows all candidate planes for user selection."""
    geometries = []
    background_pcd = o3d.geometry.PointCloud(full_pcd)
    background_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(background_pcd)

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    print("\n--- Candidate Plane Legend ---")
    for idx, (_, inliers) in enumerate(candidates):
        color = colors[idx % len(colors)]
        print(f"  [{idx}] -> Color {idx+1}")

        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(inliers)
        plane_pcd.paint_uniform_color(color)
        geometries.append(plane_pcd)

    o3d.visualization.draw_geometries(
        geometries, window_name="Select the plane that corresponds to the AprilTag"
    )


# --- MAIN SCRIPT ---
def main():
    frames = list_paired_frames(RGB_FOLDER, PCD_FOLDER)
    print(f"Found {len(frames)} frame pairs\n")

    correspondences = []

    for rgb_path, pcd_path, name in frames:
        print("─" * 60)
        print(f"Processing Frame: {name}")

        # 1. Get Camera Plane
        img = cv2.imread(rgb_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cam_plane_eq = detect_apriltag_pose(gray)
        if cam_plane_eq is None:
            print("  > AprilTag not found in image. Skipping.")
            continue
        print("  > ✅ AprilTag found in camera.")

        # 2. Get LiDAR Plane (Interactively)
        pcd = o3d.io.read_point_cloud(pcd_path)
        candidate_planes = find_planes_in_manual_region(pcd)

        if not candidate_planes:
            print("  > No cube-sized planes found in your selected region. Skipping.")
            continue

        # 3. User selects the correct plane
        visualize_candidate_planes(pcd, candidate_planes)

        try:
            sel = input(
                "Enter the index of the correct plane (or press Enter to skip): "
            ).strip()
            if not sel:
                print("  > Frame skipped by user.")
                continue
            idx = int(sel)
            if not (0 <= idx < len(candidate_planes)):
                print("  > Invalid index. Skipping.")
                continue
        except ValueError:
            print("  > Invalid input. Skipping.")
            continue

        lidar_plane_eq = candidate_planes[idx][0]
        n_c, d_c = cam_plane_eq[:3], cam_plane_eq[3]
        n_l, d_l = lidar_plane_eq[:3], lidar_plane_eq[3]

        correspondences.append((n_c, d_c, n_l, d_l))
        print(f"  > ✅ Correspondence pair #{len(correspondences)} collected!")

    if len(correspondences) < 3:
        print(
            f"\nERROR: Need at least 3 correspondences, but only found {len(correspondences)}."
        )
        return

    # --- Solve for the transformation ---
    Nc = np.stack([c[0] for c in correspondences], axis=1)
    Nl = np.stack([c[2] for c in correspondences], axis=1)
    H = Nc @ Nl.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    A = np.stack([c[2] for c in correspondences], axis=0)
    b = np.array([c[1] - c[3] for c in correspondences])
    t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = t

    print("\n==== Calibration Result ====")
    print("Final Extrinsic Matrix (Camera-to-LiDAR):")
    print(np.round(extrinsic_matrix, 6))
    np.savetxt(SAVE_MATRIX, extrinsic_matrix)
    print(f"Saved calibration matrix to {SAVE_MATRIX}")


if __name__ == "__main__":

    main()
