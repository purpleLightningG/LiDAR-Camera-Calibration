#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import open3d as o3d
from pyapriltags import Detector
from sklearn.cluster import DBSCAN

# NEW: for refinement
from scipy.optimize import least_squares  # NEW

# ─── USER SETTINGS ─────────────────────────────────────────────────────────────
RGB_FOLDER = r"folder/to/rgb"
PCD_FOLDER = r"folder/to/pcd"
TAG_ID = 8
TAG_SIZE = 0.2  # meters

# Camera intrinsics (pinhole K) — SAME device used for capture
FX, FY = 935.93473508, 933.21525291
CX, CY = 965.03666587, 555.5771423
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=float)

# Distortion coeffs (OpenCV rational model order: k1,k2,p1,p2,k3,k4,k5,k6)
distCoeffs = np.array(
    [0.11554320, -0.12274657, 0.00286100, 0.00252501, 0.07627856, 0, 0, 0], dtype=float
)

# LiDAR plane extraction (keep your RANSAC the same)
DBSCAN_EPS = 0.04
DBSCAN_MIN_SAM = 30
RANSAC_DIST = 0.01
RANSAC_N = 3
RANSAC_ITERS = 1500

# Face size gate for LiDAR plane (target ~31.3 cm × 29.2 cm)
FACE_W = 0.313  # meters
FACE_H = 0.292  # meters
FACE_TOL = 0.030  # ±3 cm tolerance per side

# LiDAR→Camera axis transform (Xᵢ→Z_c, Yᵢ→–X_c, Zᵢ→–Y_c) for visualization/extraction
M_L2C = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)

# NEW: optional refinement knobs
USE_REFINEMENT = True  # turn on global LM refinement
BASELINE_PRIOR_M = 0  # meters (set 0 to disable)
BASELINE_PRIOR_SIGMA = 0.05  # meters, strength of the prior
ANGLE_DEG_MAX = 70.0  # drop frames with alpha > this (too grazing)
MIN_PX_SIDE = 60.0  # drop frames if tag edge in pixels below this
HUBER_DELTA_ANG_DEG = 2.0  # robustify angular residuals (deg)
HUBER_DELTA_OFF_M = 0.01  # robustify offset residuals (m)
# ────────────────────────────────────────────────────────────────────────────────


def list_paired_frames(rgb_folder, pcd_folder):
    rgbs = [
        os.path.splitext(f)[0]
        for f in os.listdir(rgb_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]
    pcds = [
        os.path.splitext(f)[0]
        for f in os.listdir(pcd_folder)
        if f.lower().endswith(".pcd")
    ]
    common = sorted(set(rgbs) & set(pcds))
    pairs = []
    for b in common:
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            img_path = os.path.join(rgb_folder, b + ext)
            if os.path.exists(img_path):
                pairs.append((img_path, os.path.join(pcd_folder, b + ".pcd"), b))
                break
    return pairs


def detect_apriltag(gray):
    # 1) CLAHE for better local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # 2) Mild blur to smooth sensor noise
    gray_f = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # 3) Primary pass
    det = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )
    tags = det.detect(gray_f, estimate_tag_pose=False)
    del det

    # 4) Fallback if nothing found
    if not tags:
        det2 = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=2.0,
            quad_sigma=0.5,
            refine_edges=1,
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
    s = tag_size / 2.0
    objp = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float64)
    obj2d = objp[:, :2].astype(np.float64)
    Kinv = np.linalg.inv(K)

    idxs = [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
        [0, 3, 2, 1],
        [3, 2, 1, 0],
        [2, 1, 0, 3],
        [1, 0, 3, 2],
    ]

    best = None
    for id4 in idxs:
        imgp = corners_ud[id4].astype(np.float64)

        H = cv2.getPerspectiveTransform(
            obj2d.astype(np.float32), imgp.astype(np.float32)
        )  # 3x3 homography
        B = Kinv @ H
        h1, h2, h3 = B[:, 0], B[:, 1], B[:, 2]
        lam = 1.0 / np.linalg.norm(h1)
        r1 = lam * h1
        r2 = lam * h2
        r3 = np.cross(r1, r2)
        R = np.column_stack((r1, r2, r3))
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        t = lam * h3

        # reprojection RMS (no distortion; points are undistorted)
        P = K @ np.column_stack((R, t))
        obj_h = np.column_stack(
            (objp[:, :2], np.zeros((4, 1)), np.ones(4))
        )  # [X Y 0 1]
        proj_h = (P @ obj_h.T).T
        proj = (proj_h[:, :2].T / proj_h[:, 2]).T
        err = imgp - proj
        rms = float(np.sqrt((err**2).sum(axis=1).mean()))

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
    map1, map2 = cv2.initUndistortRectifyMap(
        K, distCoeffs, None, K, (w, h), cv2.CV_16SC2
    )
    return map1, map2


def sequential_ransac_planes(points, threshold, min_inliers, max_planes, iterations):
    all_planes = []
    rem = points.copy()
    for _ in range(max_planes):
        if len(rem) < min_inliers:
            break
        best_inl, best_params = [], None
        for __ in range(iterations):
            samp = rem[np.random.choice(len(rem), 3, replace=False)]
            cen = samp.mean(axis=0)
            _, _, vh = np.linalg.svd(samp - cen)
            n = vh[2]
            d = -n.dot(cen)
            dist = np.abs((rem @ n) + d) / (np.linalg.norm(n) + 1e-12)
            inl = rem[dist < threshold]
            if len(inl) > len(best_inl):
                best_inl, best_params = inl, (n / np.linalg.norm(n), d)
        if len(best_inl) < min_inliers:
            break
        all_planes.append((best_params, best_inl))
        n, d = best_params
        rem = rem[np.abs((rem @ n) + d) >= threshold]
    return all_planes


def extract_candidate_planes(pts):
    print(f" ↳ DBSCAN(eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAM})…")
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAM).fit_predict(pts)
    clusters = [pts[labels == label_id] for label_id in set(labels) if label_id != -1]
    print(f" • {len(clusters)} clusters")

    cands = []
    for i, cl in enumerate(clusters):
        print(f"  Cluster{i}: {len(cl)} pts", end=" → ")
        if len(cl) < 30:
            print("skip")
            continue
        planes = sequential_ransac_planes(cl, RANSAC_DIST, 30, 3, RANSAC_ITERS)
        print(f"{len(planes)} planes")
        for (n, d), inl in planes:
            cands.append((n, d, inl))
    return cands



# --- Robust face-size estimation on a plane (convex hull + minAreaRect + PCA fallback)
def face_extents_minarect(inliers_xyz, plane_normal):
    """
    Returns (short, long, thickness, debug) where short/long are face side lengths (m)
    in the plane, and thickness is spread along normal.
    """
    inliers_xyz = np.asarray(inliers_xyz, dtype=np.float64)
    if inliers_xyz.shape[0] < 10:
        return 0.0, 0.0, 0.0, {"reason": "too_few_points"}

    n = np.asarray(plane_normal, dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12

    # Stable plane basis
    axis = np.array([1.0, 0.0, 0.0])
    if abs(n @ axis) > 0.9:
        axis = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, axis)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)

    Pu = inliers_xyz @ u
    Pv = inliers_xyz @ v
    P2 = np.stack([Pu, Pv], axis=1).astype(np.float32)

    spread_u = float(Pu.max() - Pu.min())
    spread_v = float(Pv.max() - Pv.min())

    # unique 2D points for a stable hull
    P2_unique = np.unique(P2.view([("", P2.dtype)] * 2)).view(P2.dtype).reshape(-1, 2)
    if P2_unique.shape[0] < 4 or (spread_u < 1e-4 and spread_v < 1e-4):
        dn = inliers_xyz @ n
        thickness = float(dn.max() - dn.min())
        return (
            0.0,
            0.0,
            thickness,
            {"reason": "degenerate_2d", "spread_u": spread_u, "spread_v": spread_v},
        )

    hull = cv2.convexHull(P2_unique)
    rect = cv2.minAreaRect(hull)
    w, h = rect[1]
    short, long = (min(w, h), max(w, h))

    # thickness along normal
    dn = inliers_xyz @ n
    thickness = float(dn.max() - dn.min())

    # PCA fallback if rect collapses
    if short < 1e-6 and long < 1e-6:
        C = P2_unique - P2_unique.mean(axis=0, keepdims=True)
        # eigen-extents via SVD
        U, S, VT = np.linalg.svd(C, full_matrices=False)
        a = C @ VT[0]
        b = C @ VT[1]
        e1 = float(a.max() - a.min())
        e2 = float(b.max() - b.min())
        short, long = (min(e1, e2), max(e1, e2))

    dbg = {
        "spread_u": spread_u,
        "spread_v": spread_v,
        "unique": int(P2_unique.shape[0]),
    }
    return float(short), float(long), thickness, dbg


def visualize_all_planes(full_pcd, cands):
    grey = np.tile([0.7, 0.7, 0.7], (len(full_pcd.points), 1))
    full_pcd.colors = o3d.utility.Vector3dVector(grey)

    named_colors = [
        ("red", [1, 0, 0]),
        ("green", [0, 1, 0]),
        ("blue", [0, 0, 1]),
        ("yellow", [1, 1, 0]),
        ("magenta", [1, 0, 1]),
        ("cyan", [0, 1, 1]),
        ("orange", [1, 0.5, 0]),
        ("purple", [0.5, 0, 0.5]),
        ("brown", [0.6, 0.3, 0]),
        ("lavender", [0.9, 0.9, 0.98]),
        ("turquoise", [0.25, 0.88, 0.82]),
        ("khaki", [0.94, 0.9, 0.55]),
        ("maroon", [0.5, 0, 0]),
        ("gold", [1, 0.84, 0]),
        ("salmon", [0.98, 0.5, 0.45]),
        ("coral", [1, 0.5, 0.31]),
        ("mint", [0.6, 1, 0.6]),
        ("indigo", [0.29, 0, 0.51]),
        ("pink", [1, 0.75, 0.8]),
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
                o3d.utility.Vector3dVector(inl)
            )
            obb.color = rgb
            vis.add_geometry(obb)
            label_pos = obb.get_center()
        except Exception:
            label_pos = inl.mean(axis=0)

        try:
            vis.add_3d_label(label_pos, f"{idx}:{name}")
        except Exception:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sph.translate(label_pos)
            sph.paint_uniform_color(rgb)
            vis.add_geometry(sph)

    vis.run()
    vis.destroy_window()


# NEW: small helpers for SO(3) updates
def _hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=float)


def _exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = _hat(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _huber(r, delta):
    a = np.abs(r)
    w = np.ones_like(r)
    idx = a > delta
    w[idx] = delta / (a[idx] + 1e-12)
    return r * w  # reweighted residual


def calibrate():
    corr = []
    frames = list_paired_frames(RGB_FOLDER, PCD_FOLDER)
    print(f"Found {len(frames)} frame pairs\n")

    # persistent camera preview window
    win_name = "Camera plane (homography) — undistorted K view"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    kept = 0  # NEW: count kept frames

    for rgb_p, pcd_p, name in frames:
        print("─" * 60, f"\nFrame {name}")
        img = cv2.imread(rgb_p, cv2.IMREAD_COLOR)
        if img is None:
            print(" ⚠️ bad RGB")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners_raw = detect_apriltag(gray)
        if corners_raw is None:
            print(" (no tag)")
            continue
        corners_ud = undistort_corners_to_K(corners_raw)

        # --- PRINT TAG CORNERS (raw + undistorted) -------------------------------
        print("\n[Tag corners dump]")
        print("corners_raw (pixel coords in RAW image):")
        for x, y in corners_raw:
            print(f"  [{x:.2f}, {y:.2f}],")

        print("corners_ud (undistorted pixel coords in K domain):")
        print("corners_ud = [")
        for x, y in corners_ud:
            print(f"  [{x:.2f}, {y:.2f}],")
        print("]\n")

        # Camera plane from homography (you kept this approach)
        n_c, d_c, rms, R_vis, t_vis = homography_pose_from_corners(
            corners_ud, K, TAG_SIZE, return_rms=True
        )
        Z_center = float(np.linalg.norm(t_vis))
        Z_plane = float(abs(d_c))

        # NEW: quick geometry gates (drop weak frames)
        edges = [
            np.linalg.norm(corners_ud[(i + 1) % 4] - corners_ud[i]) for i in range(4)
        ]
        px_side = float(np.mean(edges))
        alpha_deg = float(
            np.degrees(np.arccos(np.clip(n_c[2], -1, 1)))
        )  # approx tilt to optical axis
        if px_side < MIN_PX_SIDE or alpha_deg > ANGLE_DEG_MAX:
            print(
                f" dropped (px_side={px_side:.1f} < {MIN_PX_SIDE} or alpha={alpha_deg:.1f}° > {ANGLE_DEG_MAX}°)"
            )
            continue

        # Make an undistorted display image in the K domain
        map1, map2 = undistort_setup_to_K(img.shape)
        img_udK = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        draw = img_udK.copy()

        cu = corners_ud.astype(int)
        for i in range(4):
            cv2.line(draw, tuple(cu[i]), tuple(cu[(i + 1) % 4]), (0, 255, 0), 2)

        axis = np.float32(
            [[0, 0, 0], [TAG_SIZE, 0, 0], [0, TAG_SIZE, 0], [0, 0, -TAG_SIZE]]
        ).reshape(-1, 3)
        imgpts, _ = cv2.projectPoints(axis, cv2.Rodrigues(R_vis)[0], t_vis, K, None)
        p0, pX, pY, pZ = imgpts.reshape(-1, 2).astype(int)
        cv2.line(draw, tuple(p0), tuple(pX), (255, 0, 0), 2)
        cv2.line(draw, tuple(p0), tuple(pY), (0, 255, 0), 2)
        cv2.line(draw, tuple(p0), tuple(pZ), (0, 0, 255), 2)

        # Z_pix = (K[0, 0] * TAG_SIZE / px_side) if px_side > 1e-6 else float("nan")
        overlay = (
            f"{name}.jpg | H-decomp: |t|={Z_center:.3f} m   "
            f"|d|={Z_plane:.3f} m   RMS={rms:.2f} px   "
            f"px_side={px_side:.1f}px  alpha≈{alpha_deg:.1f}°"
        )
        cv2.putText(
            draw,
            overlay,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            draw,
            overlay,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        print(" [y] rgb ok  [n] skip  [q] quit")
        cv2.imshow(win_name, draw)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            print(" quitting.")
            break
        if k != ord("y"):
            print(" skipped.")
            continue
        cv2.destroyWindow(win_name)
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # load & rotate LiDAR to camera-like axis convention
        pcd = o3d.io.read_point_cloud(pcd_p)
        print(f" raw LiDAR: {len(pcd.points)} pts")
        pts = np.asarray(pcd.points)
        pts_cam = (M_L2C @ pts.T).T
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        o3d.visualization.draw_geometries([pcd])

        # extract & filter face-sized planes from LiDAR
        all_cands = extract_candidate_planes(pts_cam)
        filtered = []
        for n_l, d_l, inl in all_cands:
            short, long, thick, dbg = face_extents_minarect(inl, n_l)
            face_ok = (
                abs(short - min(FACE_W, FACE_H)) < FACE_TOL
                and abs(long - max(FACE_W, FACE_H)) < FACE_TOL
            ) or (
                abs(short - max(FACE_W, FACE_H)) < FACE_TOL
                and abs(long - min(FACE_W, FACE_H)) < FACE_TOL
            )
            thickness_ok = thick < 0.05
            if face_ok and thickness_ok and short > 1e-4 and long > 1e-4:
                filtered.append((n_l, d_l, inl))
            else:
                print(
                    f"    face check -> short={short:.3f} long={long:.3f} thick={thick:.3f}  "
                    f"spread(u,v)=({dbg.get('spread_u',0):.3f},{dbg.get('spread_v',0):.3f}) "
                    f"unique2D={dbg.get('unique',0)}  -> rejected"
                )

        print(f" → {len(filtered)} face-sized candidates")
        if not filtered:
            print(" no faces; skip\n")
            continue

        visualize_all_planes(pcd, filtered)
        sel = input("Enter idx or ‘q’ to skip frame: ").strip().lower()
        if sel in ("q", ""):
            print(" skipped.\n")
            continue
        idx = int(sel)
        n_l, d_l, inl = filtered[idx]

        # Flip LiDAR normal if opposite to camera's normal
        if np.dot(n_c, n_l) < 0:
            n_l = -n_l
            d_l = -d_l

        tag_ctr = -d_c * n_c
        lidar_ctr = inl.mean(axis=0)
        print(" cam plane center (on normal):", np.round(tag_ctr, 3))
        print(" LiDAR inlier centroid:", np.round(lidar_ctr, 3))

        corr.append((n_c, d_c, n_l, d_l))
        kept += 1
        print(f" accepted plane {idx}\n")

    if len(corr) < 3:
        print("Need ≥3 corr, got", len(corr))
        sys.exit(1)

    # Closed-form init (your existing SVD)
    Nc = np.stack([c[0] for c in corr], axis=1)  # 3×m
    Nl = np.stack([c[2] for c in corr], axis=1)  # 3×m
    H = Nl @ Nc.T
    U, _, Vt = np.linalg.svd(H)
    R0 = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt

    A = np.stack([c[2] for c in corr], axis=0)  # rows = n_l^T
    b = np.array([c[1] - c[3] for c in corr], dtype=float)  # d_c - d_l
    t0, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Pretty-print initial T (FIXED: stack shapes)
    T_init = np.eye(4)
    T_init[:3, :3] = R0
    T_init[:3, 3] = t0
    print("\ninitial extrinsic matrix [camera→LiDAR]:\n", T_init)

    # Print residual check (like your logs)
    for i, (nc, dc, nl, dl) in enumerate(corr):
        lhs = float(nl @ t0)
        rhs = float(dc - dl)
        res = lhs - rhs
        print(
            f"corr[{i}]: n_c={nc.tolist()}, d_c={dc:.4f} | n_l={nl.tolist()}, d_l={dl:.4f} "
            f" → n_l·t={lhs:.4f} vs d_c-d_l={rhs:.4f} (res={res:+.4f})"
        )

    if not USE_REFINEMENT:
        extr = T_init
        print("\nfinal extrinsic matrix [camera→LiDAR]:\n", extr)
        return

    # NEW: Nonlinear refinement of R,t with robust losses and optional baseline prior
    nc_list = [c[0] for c in corr]
    dc_list = [c[1] for c in corr]
    nl_list = [c[2] for c in corr]
    dl_list = [c[3] for c in corr]

    def pack(R, t):
        # minimal: 3 for so3 (axis-angle), 3 for t; start at zero increment
        return np.zeros(6, dtype=float)

    def unpack(x, R_ref, t_ref):
        w = x[:3]
        dt = x[3:]
        R = _exp_so3(w) @ R_ref
        t = t_ref + dt
        return R, t

    def residuals(x, R_ref, t_ref):
        R, t = unpack(x, R_ref, t_ref)
        res = []
        for nc, dc, nl, dl in zip(nc_list, dc_list, nl_list, dl_list):
            # normal alignment residual (angle, in degrees)
            v = np.clip(nl @ (R @ nc), -1.0, 1.0)
            ang = np.degrees(np.arccos(v))
            res.append(_huber(np.array([ang]), HUBER_DELTA_ANG_DEG))
            # offset residual (meters)
            off = (nl @ t) - (dc - dl)
            res.append(_huber(np.array([off]), HUBER_DELTA_OFF_M))
        # optional baseline prior on |t|
        if BASELINE_PRIOR_M > 0:
            res.append(
                np.array(
                    [(np.linalg.norm(t) - BASELINE_PRIOR_M) / BASELINE_PRIOR_SIGMA]
                )
            )
        return np.concatenate(res)

    x0 = pack(R0, t0)
    sol = least_squares(
        residuals,
        x0,
        args=(R0, t0),
        verbose=1,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
        max_nfev=200,
    )
    Rf, tf = unpack(sol.x, R0, t0)

    print(f"\nRefinement success: cost={sol.cost:.6f}, iters={sol.nfev}")
    print(f"det(R): {np.linalg.det(Rf):.8f}")
    print(f"Baseline |t| [m]: {np.linalg.norm(tf):.6f}\n")

    for i, (nc, dc, nl, dl) in enumerate(corr):
        lhs = float(nl @ tf)
        rhs = float(dc - dl)
        res = lhs - rhs
        print(
            f"corr[{i}]: n_c={nc.tolist()}, d_c={dc:.4f} | n_l={nl.tolist()}, d_l={dl:.4f}  "
            f"→ n_l·t={lhs:.4f} vs d_c-d_l={rhs:.4f} (res={res:+.4f}) "
        )

    extr = np.eye(4)
    extr[:3, :3] = Rf
    extr[:3, 3] = tf
    print("\nfinal extrinsic matrix [camera→LiDAR] (refined):\n", extr)


if __name__ == "__main__":
    calibrate()


