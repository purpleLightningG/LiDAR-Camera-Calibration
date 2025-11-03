#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import open3d as o3d
from pyapriltags import Detector

# ─── USER SETTINGS (FOR SMALL TAG DATASET) ──────────────────────────────────
RGB_FOLDER    = r"folder/to/rgb"
PCD_FOLDER    = r"folder/to/pcd"
TAG_ID        = 8
TAG_SIZE      = 0.081   # meters (outer black square size)

# Camera intrinsics (pinhole K) - Using your latest high-quality calibration
FX, FY = 935.93473508, 933.21525291
CX, CY = 965.03666587, 555.5771423
K = np.array([[FX,  0, CX],
              [ 0, FY, CY],
              [ 0,  0,  1]], dtype=float)

# Distortion coeffs (OpenCV rational) - Using your latest high-quality calibration
distCoeffs = np.array([
    0.11554320, -0.12274657, 0.00286100,
    0.00252501, 0.07627856, 0,
    0, 0
], dtype=float)

# LiDAR plane extraction parameters
RANSAC_DIST    = 0.01
RANSAC_N       = 3
RANSAC_ITERS   = 1000
MIN_PLANE_INLIERS = 10 # A smaller tag will have fewer points

# LiDAR→Camera axis transform (Xᵢ→Z_c, Yᵢ→–X_c, Zᵢ→–Y_c)
M_L2C = np.array([[ 0, -1,  0],
                  [ 0,  0, -1],
                  [ 1,  0,  0]], dtype=float)
# ────────────────────────────────────────────────────────────────────────────────


def list_paired_frames(rgb_folder, pcd_folder):
    rgbs = [os.path.splitext(f)[0] for f in os.listdir(rgb_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    pcds = [os.path.splitext(f)[0] for f in os.listdir(pcd_folder) if f.lower().endswith('.pcd')]
    common = sorted(set(rgbs) & set(pcds))
    pairs = []
    for b in common:
        for ext in ('.png', '.jpg', '.jpeg'):
            img_path = os.path.join(rgb_folder, b+ext)
            if os.path.exists(img_path):
                pairs.append((img_path, os.path.join(pcd_folder, b+'.pcd'), b))
                break
    return pairs

def detect_apriltag(gray):
    det = Detector(families='tag36h11', nthreads=4, quad_decimate=1.0, refine_edges=1)
    tags = det.detect(gray, estimate_tag_pose=False)
    for t in tags:
        if t.tag_id == TAG_ID:
            return np.array(t.corners, dtype=np.float64)
    return None

def undistort_corners(corners_raw, K, dist):
    pts = corners_raw.reshape(1, -1, 2).astype(np.float64)
    return cv2.undistortPoints(pts, K, dist, P=K).reshape(4, 2)

def get_camera_plane_and_pose(corners_ud, K, tag_size):
    """Returns the plane equation (n,d) and the pose (R,t)"""
    s = tag_size/2.0
    objp = np.array([[-s,-s,0],[s,-s,0],[s,s,0],[-s,s,0]], dtype=np.float32)
    ret, rvec, tvec = cv2.solvePnP(objp, corners_ud.astype(np.float32), K, distCoeffs[:5])
    if not ret: return None, None, None, None
    R, _ = cv2.Rodrigues(rvec)
    n_c = R[:, 2]
    d_c = -n_c @ tvec.flatten()
    return n_c, d_c, R, tvec.flatten()

def project_points(pts_3d, K, T_l2c):
    R, t = T_l2c[:3,:3], T_l2c[:3,3]
    pts_cam = (R @ pts_3d.T + t.reshape(3,1)).T
    pts_proj = (K @ pts_cam.T).T
    depths = pts_proj[:,2]
    valid_mask = depths > 1e-3
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    pts_2d = (pts_proj[valid_mask, :2].T / depths[valid_mask]).T
    return pts_2d, valid_mask

def find_lidar_plane_guided(pts_lidar, T_l2c_initial, corners_ud, K, img_for_debug, frame_name):
    """
    Projects all LiDAR points, keeps only those inside an expanded bounding
    box around the tag, and then runs RANSAC on that subset.
    """
    img_pts, valid_mask = project_points(pts_lidar, K, T_l2c_initial)
    if img_pts.size == 0:
        print("    • No LiDAR points projected into camera view.")
        return None, None
        
    pts_lidar_in_view = pts_lidar[valid_mask]
    
    x, y, w, h = cv2.boundingRect(corners_ud.astype(np.int32))
    padding_w, padding_h = int(w * 0.5), int(h * 0.5)
    x1, y1 = x - padding_w, y - padding_h
    x2, y2 = x + w + padding_w, y + h + padding_h
    
    in_tag_mask = (img_pts[:, 0] >= x1) & (img_pts[:, 0] <= x2) & \
                  (img_pts[:, 1] >= y1) & (img_pts[:, 1] <= y2)

    pts_on_target = pts_lidar_in_view[in_tag_mask]
    print(f"    • Filtered to {len(pts_on_target)} LiDAR points within expanded search area.")

    debug_overlay = img_for_debug.copy()
    cv2.rectangle(debug_overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
    for pt in img_pts[in_tag_mask]:
        cv2.circle(debug_overlay, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
    cv2.imwrite(f"debug_overlay_{frame_name}.jpg", debug_overlay)
    print(f"    • Saved debug overlay to debug_overlay_{frame_name}.jpg")

    if len(pts_on_target) < MIN_PLANE_INLIERS:
        print("    -> Not enough points on target to find a plane.")
        return None, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_on_target)
    plane_model, inlier_indices = pcd.segment_plane(RANSAC_DIST, RANSAC_N, RANSAC_ITERS)
    
    if len(inlier_indices) < MIN_PLANE_INLIERS:
        print("    -> RANSAC failed to find a stable plane in the filtered points.")
        return None, None

    return plane_model[:3], plane_model[3]

def calibrate():
    corr=[]
    frames = list_paired_frames(RGB_FOLDER,PCD_FOLDER)
    print(f"Found {len(frames)} frame pairs\n")
    
    for rgb_p, pcd_p, name in frames:
        print("─"*60, f"\nFrame {name}")
        img = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        if img is None: print(" ⚠️ bad RGB"); continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners_raw = detect_apriltag(gray)
        if corners_raw is None: print(" (no tag)"); continue
        corners_ud = undistort_corners(corners_raw, K, distCoeffs)
        
        n_c, d_c, R_cam, t_cam = get_camera_plane_and_pose(corners_ud, K, TAG_SIZE)
        if n_c is None: print(" (cam pose failed)"); continue
        print("    • Camera plane found successfully.")

        pcd = o3d.io.read_point_cloud(pcd_p); 
        pts = np.asarray(pcd.points)
        print(f"    • Loaded {len(pts)} LiDAR points.")

        lidar_centroid = pts.mean(axis=0)
        lidar_centroid_cam_basis = M_L2C @ lidar_centroid
        initial_t = t_cam - lidar_centroid_cam_basis
        T_initial_guess = np.eye(4)
        T_initial_guess[:3,:3] = M_L2C
        T_initial_guess[:3,3] = initial_t
        print(f"    • Generated initial guess for translation: {np.round(initial_t, 3)}")

        n_l, d_l = find_lidar_plane_guided(pts, T_initial_guess, corners_ud, K, img, name)
        
        if n_l is None:
            print("    -> FAILED to find a corresponding LiDAR plane. Skipping frame.")
            continue
        print("    • LiDAR plane found successfully using camera-guided search.")
        
        if np.dot(n_c, M_L2C @ n_l) < 0:
            n_l, d_l = -n_l, -d_l

        corr.append((n_c, d_c, n_l, d_l))
        print(f"    -> SUCCESS: Correspondence pair collected for frame {name}.\n")

    if len(corr) < 3:
        print(f"\nNeed at least 3 valid correspondences to solve, but only got {len(corr)}. Aborting.")
        sys.exit(1)

    print("\n" + "="*60)
    print(f"SOLVING EXTRINSICS USING {len(corr)} CORRESPONDENCE PAIRS")
    print("="*60)

    Nc = np.stack([c[0] for c in corr], axis=1)
    Nl_raw = np.stack([c[2] for c in corr], axis=1)
    Nl_basis = M_L2C @ Nl_raw 

    H = Nl_basis @ Nc.T
    U,_,Vt = np.linalg.svd(H)
    R_cam_to_lidar_basis = U @ np.diag([1,1,np.linalg.det(U@Vt)]) @ Vt
    
    R_final = M_L2C.T @ R_cam_to_lidar_basis

    A = np.stack([c[2] for c in corr], axis=0)
    b = np.array([c[1] - c[3] for c in corr], dtype=float)
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    
    extr = np.eye(4)
    extr[:3,:3] = R_final
    extr[:3, 3] = t
    print("\nFINAL EXTRINSIC MATRIX [camera→LiDAR]:\n", np.round(extr, 6))

if __name__=="__main__":
    calibrate()



