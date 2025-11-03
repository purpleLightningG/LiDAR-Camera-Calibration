#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import open3d as o3d
from pyapriltags import Detector
from sklearn.cluster import DBSCAN

# ─── USER SETTINGS ─────────────────────────────────────────────────────────────
RGB_FOLDER    =
PCD_FOLDER    = 
TAG_ID        = 8
TAG_SIZE      = 0.081   # meters (outer black square size)

# Camera intrinsics (pinhole K) — SAME device used for capture
FX, FY = 935.46323691, 932.71987717
CX, CY = 965.81127117, 555.27376472
K = np.array([[FX,  0, CX],
              [ 0, FY, CY],
              [ 0,  0,  1]], dtype=float)

# K = np.array([
#     [FX=935.93473508, 0, CX=965.03666587],
#     [0, FY=933.21525291, CY=555.5771423],
#     [0, 0, 1]
# ], dtype=float)

# Distortion coeffs (OpenCV rational model order: k1,k2,p1,p2,k3,k4,k5,k6)
distCoeffs = np.array([
    0.11491258, -0.11983334, 0.00270534,
    0.00295228, 0.07373178, 0,
    0, 0
], dtype=float)

# LiDAR plane extraction
DBSCAN_EPS     = 0.03
DBSCAN_MIN_SAM = 10
RANSAC_DIST    = 0.01
RANSAC_N       = 5
RANSAC_ITERS   = 1000

# Cube face size gate (to keep only compact planes)
CUBE_SIDE = 0.11   # meters (≈ cube face size)
CUBE_TOL  = 0.01   # ±1 cm

# LiDAR→Camera axis transform (Xᵢ→Z_c, Yᵢ→–X_c, Zᵢ→–Y_c) for visualization/extraction
M_L2C = np.array([[ 0, -1,  0],
                  [ 0,  0, -1],
                  [ 1,  0,  0]], dtype=float)
# ────────────────────────────────────────────────────────────────────────────────


def list_paired_frames(rgb_folder, pcd_folder):
    rgbs = [os.path.splitext(f)[0] for f in os.listdir(rgb_folder)
            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
    pcds = [os.path.splitext(f)[0] for f in os.listdir(pcd_folder)
            if f.lower().endswith('.pcd')]
    common = sorted(set(rgbs) & set(pcds))
    pairs = []
    for b in common:
        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            img_path = os.path.join(rgb_folder, b+ext)
            if os.path.exists(img_path):
                pairs.append((img_path, os.path.join(pcd_folder, b+'.pcd'), b))
                break
    return pairs


def detect_apriltag(gray):
    # 1) CLAHE for better local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)

    # 2) Mild blur to smooth sensor noise
    gray_f = cv2.GaussianBlur(gray_eq, (3,3), 0)

    # 3) Primary pass
    det = Detector(
        families='tag36h11',
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )
    tags = det.detect(gray_f, estimate_tag_pose=False)
    del det

    # 4) Fallback if nothing found
    if not tags:
        det2 = Detector(
            families='tag36h11',
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.5,
            refine_edges=1
        )
        tags = det2.detect(gray_f, estimate_tag_pose=False)
        del det2

    # 5) return corners of requested ID
    for t in tags:
        if t.tag_id == TAG_ID:
            return np.array(t.corners, dtype=np.float64)

    return None


def undistort_corners_to_K(corners_raw):
    """
    Undistort 4 tag corners into pixel coordinates under the SAME K (pinhole).
    Using undistortPoints with P=K returns pixel coords in the K domain.
    """
    pts = corners_raw.reshape(1, -1, 2).astype(np.float64)
    pts_ud = cv2.undistortPoints(pts, K, distCoeffs, P=K)
    return pts_ud.reshape(4, 2)


def homography_pose_from_corners(corners_ud, K, tag_size, return_rms=True):
    """
    Compute plane pose (R,t) from 4 undistorted pixel corners using homography.
    Try 8 permutations (4 rotations × 2 windings); select lowest reprojection RMS.

    Returns: n_c (3,), d_c (scalar), (optional) rms
       plane in camera frame:  n_c · X + d_c = 0,  with n_c = r3 = R[:,2],  d_c = -n_c^T t
    """
    s = tag_size/2.0
    objp = np.array([[-s,-s,0],
                     [ s,-s,0],
                     [ s, s,0],
                     [-s, s,0]], dtype=np.float64)
    obj2d = objp[:, :2].astype(np.float64)
    Kinv  = np.linalg.inv(K)

    idxs = [
        [0,1,2,3], [1,2,3,0], [2,3,0,1], [3,0,1,2],
        [0,3,2,1], [3,2,1,0], [2,1,0,3], [1,0,3,2]
    ]

    best = None
    for id4 in idxs:
        imgp = corners_ud[id4].astype(np.float64)

        H = cv2.getPerspectiveTransform(obj2d.astype(np.float32),
                                        imgp.astype(np.float32))  # 3x3 homography
        B = Kinv @ H
        h1, h2, h3 = B[:,0], B[:,1], B[:,2]
        lam = 1.0 / np.linalg.norm(h1)
        r1  = lam * h1
        r2  = lam * h2
        r3  = np.cross(r1, r2)
        R   = np.column_stack((r1, r2, r3))
        if np.linalg.det(R) < 0:
            R[:,2] *= -1
        t   = lam * h3

        # reprojection RMS (no distortion; points are undistorted)
        P = K @ np.column_stack((R, t))
        obj_h = np.column_stack((objp[:,:2], np.zeros((4,1)), np.ones(4)))  # [X Y 0 1]
        proj_h = (P @ obj_h.T).T
        proj   = (proj_h[:, :2].T / proj_h[:, 2]).T
        err    = imgp - proj
        rms    = float(np.sqrt((err**2).sum(axis=1).mean()))

        if (best is None) or (rms < best[0]):
            best = (rms, R, t)

    rms_best, R_best, t_best = best
    n_c = R_best[:, 2] / np.linalg.norm(R_best[:, 2])
    d_c = -float(n_c @ t_best)

    if return_rms:
        return n_c, d_c, rms_best, R_best, t_best
    else:
        return n_c, d_c


def undistort_setup_to_K(img_shape):
    h, w = img_shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(K, distCoeffs, None, K, (w, h), cv2.CV_16SC2)
    return map1, map2


def sequential_ransac_planes(points, threshold, min_inliers, max_planes, iterations):
    all_planes = []
    rem = points.copy()
    for _ in range(max_planes):
        if len(rem) < min_inliers:
            break
        best_inl, best_params = [], None
        for __ in range(iterations):
            samp = rem[np.random.choice(len(rem),3,replace=False)]
            cen = samp.mean(axis=0)
            _,_,vh = np.linalg.svd(samp-cen)
            n = vh[2]; d = -n.dot(cen)
            dist = np.abs((rem @ n) + d) / np.linalg.norm(n)
            inl = rem[dist < threshold]
            if len(inl) > len(best_inl):
                best_inl, best_params = inl, (n/np.linalg.norm(n), d)
        if len(best_inl) < min_inliers:
            break
        all_planes.append((best_params, best_inl))
        n, d = best_params
        rem = rem[np.abs((rem @ n) + d) >= threshold]
    return all_planes


def extract_candidate_planes(pts):
    print(f"  ↳ DBSCAN(eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAM})…")
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAM).fit_predict(pts)
    clusters = [pts[labels==l] for l in set(labels) if l != -1]
    print(f"    • {len(clusters)} clusters")
    cands=[]
    for i,cl in enumerate(clusters):
        print(f"    Cluster{i}: {len(cl)} pts", end=" → ")
        if len(cl) < 50:
            print("skip"); continue
        planes = sequential_ransac_planes(cl, RANSAC_DIST, 10, 3, RANSAC_ITERS)
        print(f"{len(planes)} planes")
        for (n,d),inl in planes:
            cands.append((n,d,inl))
    return cands


def visualize_all_planes(full_pcd, cands):
    grey = np.tile([0.7, 0.7, 0.7], (len(full_pcd.points), 1))
    full_pcd.colors = o3d.utility.Vector3dVector(grey)

    named_colors = [
        ("red",      [1, 0, 0]), ("green",    [0, 1, 0]),
        ("blue",     [0, 0, 1]), ("yellow",   [1, 1, 0]),
        ("magenta",  [1, 0, 1]), ("cyan",     [0, 1, 1]),
        ("orange",   [1, 0.5, 0]), ("purple", [0.5, 0, 0.5]),
        ("brown",    [0.6, 0.3, 0]), 
        ("lavender", [0.9, 0.9, 0.98]), ("turquoise",[0.25, 0.88, 0.82]),
        ("khaki",    [0.94, 0.9, 0.55]), ("maroon",[0.5, 0, 0]),
        ("gold",     [1, 0.84, 0]), ("salmon",[0.98, 0.5, 0.45]),
        ("coral",    [1, 0.5, 0.31]), ("mint",[0.6, 1, 0.6]),
        ("indigo",   [0.29, 0, 0.51]), ("pink", [1, 0.75, 0.8]),
    ]

    print("\n--- Plane legend (idx → color) ---")
    for idx, _ in enumerate(cands):
        name, _ = named_colors[idx % len(named_colors)]
        print(f"  [{idx}] → {name}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Plane Candidates")
    vis.add_geometry(full_pcd)

    for idx, (_, _, inl) in enumerate(cands):
        name, rgb = named_colors[idx % len(named_colors)]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(inl)
        pc.paint_uniform_color(rgb)
        vis.add_geometry(pc)

        try:
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(inl))
            obb.color = rgb
            vis.add_geometry(obb)
            label_pos = obb.get_center()
        except:
            label_pos = inl.mean(axis=0)

        try:
            vis.add_3d_label(label_pos, f"{idx}:{name}")
        except AttributeError:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sph.translate(label_pos)
            sph.paint_uniform_color(rgb)
            vis.add_geometry(sph)

    vis.run()
    vis.destroy_window()


def calibrate():
    corr=[]
    frames = list_paired_frames(RGB_FOLDER,PCD_FOLDER)
    print(f"Found {len(frames)} frame pairs\n")

    # persistent camera preview window
    win_name = "Camera plane (homography) — undistorted K view"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    for rgb_p, pcd_p, name in frames:
        print("─"*60, f"\nFrame {name}")
        img = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        if img is None:
            print(" ⚠️ bad RGB"); continue

        # Detect tag on RAW, then undistort corners into SAME K domain
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners_raw = detect_apriltag(gray)
        if corners_raw is None:
            print(" (no tag)"); continue
        corners_ud = undistort_corners_to_K(corners_raw)

        # Plane (camera) from homography (no PnP)
        n_c, d_c, rms, R_vis, t_vis = homography_pose_from_corners(corners_ud, K, TAG_SIZE, return_rms=True)
        Z_center = float(np.linalg.norm(t_vis))
        Z_plane  = float(abs(d_c))

        # Make an undistorted display image in the K domain
        map1, map2 = undistort_setup_to_K(img.shape)
        img_udK = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        draw = img_udK.copy()

        # draw polygon at undistorted corners
        cu = corners_ud.astype(int)
        for i in range(4):
            cv2.line(draw, tuple(cu[i]), tuple(cu[(i+1) % 4]), (0,255,0), 2)

        # draw axes for visualization (project with K, no distortion)
        axis = np.float32([[0,0,0],
                           [TAG_SIZE,0,0],
                           [0,TAG_SIZE,0],
                           [0,0,-TAG_SIZE]]).reshape(-1,3)
        imgpts, _ = cv2.projectPoints(axis, cv2.Rodrigues(R_vis)[0], t_vis, K, None)
        p0, pX, pY, pZ = imgpts.reshape(-1,2).astype(int)
        cv2.line(draw, tuple(p0), tuple(pX), (255,0,0), 2)
        cv2.line(draw, tuple(p0), tuple(pY), (0,255,0), 2)
        cv2.line(draw, tuple(p0), tuple(pZ), (0,0,255), 2)

        # quick pixel-based range from undistorted corners
        edges = [np.linalg.norm(corners_ud[(i+1)%4]-corners_ud[i]) for i in range(4)]
        px_side = float(np.mean(edges))
        Z_pix = (K[0,0] * TAG_SIZE / px_side) if px_side > 1e-6 else float('nan')

        # ONE-LINE OVERLAY (exact format requested)
        overlay = (f"{name}.jpg | H-decomp: |t|={Z_center:.3f} m   "
                   f"|d|={Z_plane:.3f} m   RMS={rms:.2f} px   "
                   f"Pixel_ud={Z_pix:.3f} m")
        cv2.putText(draw, overlay, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(draw, overlay, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        # show and ask for confirmation
        print(" [y] rgb ok  [n] skip  [q] quit")
        cv2.imshow(win_name, draw)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            print(" quitting."); break
        if k != ord('y'):
            print(" skipped."); continue
        cv2.destroyWindow(win_name)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # reopen clean for next frame

        # load & rotate LiDAR to camera-like axis convention (for plane extraction)
        pcd = o3d.io.read_point_cloud(pcd_p)
        print(f" raw LiDAR: {len(pcd.points)} pts")
        pts = np.asarray(pcd.points)
        pts_cam = (M_L2C @ pts.T).T
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        o3d.visualization.draw_geometries([pcd])

        # extract & filter cube‐sized planes from LiDAR
        all_cands = extract_candidate_planes(pts_cam)
        filtered=[]
        for n_l,d_l,inl in all_cands:
            try:
                obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(inl))
                if max(obb.extent) < CUBE_SIDE*2:
                    filtered.append((n_l,d_l,inl))
            except:
                pass
        print(f" → {len(filtered)} cube‐sized candidates")
        if not filtered:
            print(" no faces; skip\n"); continue

        # user picks the cube face
        visualize_all_planes(pcd, filtered)
        sel = input("Enter idx or ‘q’ to skip frame: ").strip().lower()
        if sel in ("q",""):
            print(" skipped.\n"); continue
        idx = int(sel)
        n_l, d_l, inl = filtered[idx]

        # Flip LiDAR normal if opposite to camera's normal
        if np.dot(n_c, n_l) < 0:
            n_l = -n_l
            d_l = -d_l

        # optional: centers
        tag_ctr   = -d_c * n_c
        lidar_ctr = inl.mean(axis=0)
        print(" cam plane center (on normal):", np.round(tag_ctr,3))
        print(" LiDAR inlier centroid:",       np.round(lidar_ctr,3))

        corr.append((n_c, d_c, n_l, d_l))
        print(f" accepted plane {idx}\n")

    if len(corr) < 3:
        print("Need ≥3 corr, got", len(corr)); sys.exit(1)

    # solve rotation (align LiDAR normals to camera normals)
    Nc = np.stack([c[0] for c in corr], axis=1)  # 3×m
    Nl = np.stack([c[2] for c in corr], axis=1)  # 3×m
    H  = Nl @ Nc.T
    U,_,Vt = np.linalg.svd(H)
    R = U @ np.diag([1,1,np.linalg.det(U@Vt)]) @ Vt

    # translation from plane offsets: n_l · t = d_c - d_l   (camera→LiDAR t)
    A = np.stack([c[2] for c in corr], axis=0)               # rows = n_l^T
    b = np.array([c[1] - c[3] for c in corr], dtype=float)   # d_c - d_l
    t, *_ = np.linalg.lstsq(A, b, rcond=None)

    for i,(nc,dc,nl,dl) in enumerate(corr):
        print(f"corr[{i}]: n_c={nc.tolist()}, d_c={dc:.4f} | n_l={nl.tolist()}, d_l={dl:.4f}")

    extr = np.eye(4)
    extr[:3,:3] = R
    extr[:3, 3] = t
    print("\nfinal extrinsic matrix [camera→LiDAR]:\n", extr)


if __name__=="__main__":
    calibrate()

