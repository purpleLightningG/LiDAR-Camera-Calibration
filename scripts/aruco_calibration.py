import numpy as np
import cv2
import open3d as o3d
import os

# --- CONFIGURATION ---
DATA_PATH = r"folder/to/rgb"
BOARD_WIDTH = 0.203  # A4 width in meters
BOARD_HEIGHT = 0.267  # A4 height in meters
ARUCO_ID = 0
ARUCO_SIZE = 0.14  # Aruco marker size (meters, now 14cm)

K = np.array([[912.0468, 0, 962.5100], [0, 911.8128, 546.4423], [0, 0, 1]])
DIST_COEFFS = np.zeros((4, 1))


def get_board_corners_in_object_frame(dx=0.040, dy=-0.030):
    half_w, half_h = BOARD_WIDTH / 2, BOARD_HEIGHT / 2
    return np.array(
        [
            [-half_w - dx, half_h - dy, 0],  # Top-left
            [half_w - dx, half_h - dy, 0],  # Top-right
            [half_w - dx, -half_h - dy, 0],  # Bottom-right
            [-half_w - dx, -half_h - dy, 0],  # Bottom-left
        ],
        dtype=np.float32,
    )


def get_aruco_in_image(image, manual_corners=False):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=params)
    computed_img_corners = None

    if ids is not None:
        for i, marker_id in enumerate(ids):
            if marker_id[0] == ARUCO_ID:
                c = corners[i][0]  # 4x2 array
                aruco_object_pts = np.array(
                    [
                        [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
                        [ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
                        [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
                        [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
                    ],
                    dtype=np.float32,
                )
                ret, rvec, tvec = cv2.solvePnP(aruco_object_pts, c, K, DIST_COEFFS)
                if not ret:
                    return None, None, None
                R, _ = cv2.Rodrigues(rvec)
                board_object_pts = get_board_corners_in_object_frame(
                    dx=0.040, dy=-0.030
                )
                board_corners_cam = (R @ board_object_pts.T).T + tvec.T
                # Draw ArUco and axis
                cv2.aruco.drawDetectedMarkers(image, [corners[i]], ids[i])
                cv2.drawFrameAxes(image, K, DIST_COEFFS, rvec, tvec, 0.07)
                # Draw computed board corners (green)
                img_corners, _ = cv2.projectPoints(
                    board_object_pts, rvec, tvec, K, DIST_COEFFS
                )
                computed_img_corners = img_corners.reshape(-1, 2)
                for p in computed_img_corners:
                    cv2.circle(image, tuple(np.int32(p)), 7, (0, 255, 0), -1)
                # Manual click mode
                if manual_corners:
                    clicked_points = []

                    def click_event(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            print(f"Clicked at: ({x}, {y})")
                            clicked_points.append((x, y))

                    clone = image.copy()
                    cv2.namedWindow("Camera Detection")
                    cv2.setMouseCallback("Camera Detection", click_event)
                    print(
                        "Please click the 4 actual A4 paper corners (order: TL, TR, BR, BL). Press ESC to abort."
                    )
                    while len(clicked_points) < 4:
                        cv2.imshow("Camera Detection", clone)
                        key = cv2.waitKey(20)
                        if key & 0xFF == 27:  # ESC = abort
                            print("Manual corner picking aborted (ESC pressed).")
                            cv2.destroyWindow("Camera Detection")
                            return None, None, None, True  # Frame is rejected
                    print("Clicked corners (pixels):", clicked_points)
                    for pt in clicked_points:
                        cv2.circle(clone, pt, 7, (0, 0, 255), -1)
                    for p in computed_img_corners:
                        cv2.circle(clone, tuple(np.int32(p)), 7, (0, 255, 0), -1)
                    cv2.imshow("Manual vs Computed Corners", clone)
                    cv2.waitKey(500)
                    cv2.destroyWindow("Manual vs Computed Corners")
                    return (
                        board_corners_cam,
                        computed_img_corners,
                        clicked_points,
                        False,
                    )
                else:
                    cv2.imshow("Camera Detection", image)
                    key = cv2.waitKey(1)
                    if key & 0xFF == 27:  # ESC = abort
                        print("Camera detection aborted (ESC pressed).")
                        cv2.destroyWindow("Camera Detection")
                        return None, None, None, True
                    return board_corners_cam, computed_img_corners, None, False
    return None, None, None, False


# --- RANSAC plane detection code (NumPy version) ---
def fit_plane(points):
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[2, :]
    d = -np.dot(normal, centroid)
    return np.append(normal, d)


def ransac_plane(points, threshold, iterations):
    best_inliers = []
    best_params = None
    for _ in range(iterations):
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        params = fit_plane(sample)
        distances = np.abs((points @ params[:3]) + params[3]) / np.linalg.norm(
            params[:3]
        )
        inlier_mask = distances < threshold
        inliers = points[inlier_mask]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params = params
    return best_params, np.array(best_inliers)


def show_plane_result(pcd, plane_points):
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
    plane_pcd.paint_uniform_color([0, 1, 0])
    rest_points = np.asarray(pcd.points)
    # For large clouds, this is "good enough" - show plane and all cloud
    o3d.visualization.draw_geometries(
        [pcd, plane_pcd], window_name="RANSAC Plane Detection"
    )


# MAIN SCRIPT
if __name__ == "__main__":
    image_files = sorted(
        [f for f in os.listdir(os.path.join(DATA_PATH, "rgb")) if f.endswith(".jpg")]
    )
    all_cam_corners = []
    all_lidar_corners = []
    for image_name in image_files:
        frame_number_str = os.path.splitext(image_name)[0]
        print(f"\n--- [Processing Frame #{frame_number_str}] ---")
        img_path = os.path.join(DATA_PATH, "rgb", image_name)
        image = cv2.imread(img_path)
        cam_corners, projected_corners, clicked_corners, is_rejected = (
            get_aruco_in_image(image, manual_corners=True)
        )
        if is_rejected or cam_corners is None:
            print(
                "  > ⚠️ Tag or board not detected in camera or frame rejected by user. Skipping."
            )
            continue
        print("  > ✅ Detected board in camera.")
        print(f"  > Computed corners (pixels):\n{np.round(projected_corners, 1)}")
        print(f"  > Manually clicked corners (pixels):\n{clicked_corners}")

        # LiDAR: load and RANSAC
        pcd_path = os.path.join(DATA_PATH, "lidar", f"{frame_number_str}.pcd")
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        if len(points) < 100:
            print("  > ⚠️ Not enough points in LiDAR. Skipping.")
            continue
        plane_params, plane_points = ransac_plane(
            points, threshold=0.01, iterations=400
        )
        print(
            f"  > Plane equation: {np.round(plane_params, 4)}. Inliers: {len(plane_points)}"
        )
        show_plane_result(pcd, plane_points)

        # Ask user to accept/reject LiDAR plane
        user_input = input("  > Accept detected LiDAR plane? (y/n): ").lower()
        if user_input != "y":
            print("  > Detection rejected by user. Skipping this frame.")
            continue

        # Find OBB of plane (for 4 corners)
        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
        obb = plane_pcd.get_oriented_bounding_box()
        lidar_corners = np.asarray(obb.get_box_points())
        all_cam_corners.append(cam_corners)
        all_lidar_corners.append(lidar_corners)
        input("Check detection windows, then press Enter to continue...")
        cv2.destroyAllWindows()

    if len(all_cam_corners) < 3:
        print("ERROR: Need at least 3 good pairs for calibration.")
        exit(0)

    source = np.vstack(all_lidar_corners)
    target = np.vstack(all_cam_corners)
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target)

    corres = o3d.utility.Vector2iVector(
        np.array([np.arange(len(source)), np.arange(len(source))]).T
    )
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd, tgt_pcd, corres, 0.05
    )

    print("\n--- CALIBRATION COMPLETE! ---")
    print("Your final 4x4 Extrinsic Transformation Matrix is:")
    print(np.round(result.transformation, 4))
    np.savetxt("extrinsic_calibration.txt", result.transformation)
    print("\nMatrix saved to 'extrinsic_calibration.txt'")


