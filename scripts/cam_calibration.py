# -*- coding: utf-8 -*-
"""
High-Accuracy Camera Intrinsic Calibration Script using OpenCV

Purpose:
  This script calculates a camera's intrinsic parameters (camera matrix K
  and distortion coefficients) from a set of checkerboard images. It uses
  a two-pass method with sub-pixel corner refinement and outlier rejection
  for improved accuracy.

New in this version:
  • Saves a side-by-side "Original vs Undistorted" image to:
      calibration_output/undistort_preview.png
  • (Optional) Saves per-image corner overlay snapshots to:
      calibration_output/corners/<basename>_corners.jpg
"""

import numpy as np
import cv2
import os
import glob

# ============================ CONFIG ============================

IMAGES_DIR = 
CHECKERBOARD_DIMS = (7, 10)
SQUARE_SIZE_METERS = 0.021  # adjust as needed
IMAGE_EXTENSION = 'jpg'            # 'jpg', 'png', 'bmp', etc.

# Output directory (created if missing)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'calibration_output')
CORNERS_DIR = os.path.join(OUTPUT_DIR, 'corners')
SAVE_CORNER_OVERLAYS = True  # set False if you don't want per-image corner snapshots

# ===============================================================

def _safe_imshow(win, img, delay=1):
    """Show if GUI available; ignore if running headless."""
    try:
        cv2.imshow(win, img)
        cv2.waitKey(delay)
    except cv2.error:
        pass

def main():
    print("Starting camera intrinsic calibration...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_CORNER_OVERLAYS:
        os.makedirs(CORNERS_DIR, exist_ok=True)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE_METERS

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane.

    image_paths = glob.glob(os.path.join(IMAGES_DIR, f'*.{IMAGE_EXTENSION}'))
    if not image_paths:
        print(f"❌ Error: No images found in '{IMAGES_DIR}' with extension '.{IMAGE_EXTENSION}'")
        return

    print(f"Found {len(image_paths)} images to process...")

    # --- 2. Find Corners in Each Image ---
    first_image_shape = None
    valid_image_paths = []
    for fname in image_paths:
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read {fname}, skipping.")
            continue

        if first_image_shape is None:
            first_image_shape = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria) #why 11,11
            imgpoints.append(corners2)
            valid_image_paths.append(fname)

            # Draw corners (high-res) and optionally save
            img_draw = img.copy()
            cv2.drawChessboardCorners(img_draw, CHECKERBOARD_DIMS, corners2, ret)

            # Show (scaled) if GUI available
            h, w = img_draw.shape[:2]
            scale = 800 / w if w > 800 else 1.0
            img_display = cv2.resize(img_draw, (int(w * scale), int(h * scale)))
            _safe_imshow('Image with Detected Corners', img_display, delay=100)

            if SAVE_CORNER_OVERLAYS:
                base = os.path.splitext(os.path.basename(fname))[0]
                out_path = os.path.join(CORNERS_DIR, f'{base}_corners.jpg')
                cv2.imwrite(out_path, img_draw)
        else:
            print(f"Warning: Checkerboard not found in {os.path.basename(fname)}")

    cv2.destroyAllWindows()

    if len(imgpoints) < 10:
        print(f"❌ Error: Calibration failed. Only found checkerboards in {len(imgpoints)} images.")
        return

    print(f"\nSuccessfully found corners in {len(imgpoints)} / {len(image_paths)} images.")
    print("Running initial calibration pass...")

    # --- 3. First Calibration Pass ---
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, first_image_shape[::-1], None, None)
    if not ret:
        print("❌ Error: Initial calibration failed to compute.")
        return
    print(f"Initial RMS Reprojection Error: {ret:.4f} pixels")

    # --- 4. Outlier Rejection ---
    print("\nCalculating per-image reprojection error for outlier removal...")
    per_image_errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        per_image_errors.append(error)

    median_error = np.median(per_image_errors)
    outlier_threshold = 2.0 * median_error
    print(f"Median error: {median_error:.4f} pixels. Outlier threshold: {outlier_threshold:.4f} pixels.")

    refined_objpoints, refined_imgpoints = [], []
    dropped_count = 0
    for i, error in enumerate(per_image_errors):
        if error < outlier_threshold:
            refined_objpoints.append(objpoints[i])
            refined_imgpoints.append(imgpoints[i])
        else:
            print(f"  -> Dropping '{os.path.basename(valid_image_paths[i])}' (error: {error:.4f} px)")
            dropped_count += 1

    if dropped_count > 0:
        print(f"\nDropped {dropped_count} outlier images. Running final calibration on {len(refined_imgpoints)} images...")
        ret, mtx, dist, _, _ = cv2.calibrateCamera(refined_objpoints, refined_imgpoints, first_image_shape[::-1], None, None)
    else:
        print("\nNo outliers found. Using initial calibration result.")

    # --- 6. Display and Save Final Results ---
    print("\n" + "="*50)
    print("      FINAL CAMERA INTRINSIC CALIBRATION RESULTS")
    print("="*50)

    print(f"✅ Final RMS Reprojection Error: {ret:.4f} pixels")
    print("   (A value < 0.5 is considered a good calibration)\n")

    print("Camera Matrix (K):")
    print(mtx)

    dist_5_param = dist[0][:5].flatten()
    dist_8_param = np.zeros(8)
    dist_8_param[:len(dist[0])] = dist[0].flatten()

    print("\nDistortion Coefficients (Full Rational Model):")
    labels = ["k1 (Radial)", "k2 (Radial)", "p1 (Tangential)", "p2 (Tangential)",
              "k3 (Radial)", "k4 (Rational)", "k5 (Rational)", "k6 (Rational)"]
    for i, label in enumerate(labels):
        print(f"  {label:<18}: {dist_8_param[i]:.8f}")

    output_filename = os.path.join(OUTPUT_DIR, 'camera_calibration.npz')
    np.savez(output_filename,
             camera_matrix=mtx,
             dist_coeffs_5=dist_5_param,
             dist_coeffs_8=dist_8_param,
             rms_error=ret)
    print(f"\n💾 Calibration data saved to '{output_filename}'")
    print("="*50)

    # --- 7. Visual Verification (ALSO SAVED TO DISK) ---
    print("\nCreating and saving an undistortion preview...")
    img_to_test = cv2.imread(valid_image_paths[0])  # use a valid detection image
    h, w = img_to_test.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img_to_test, mtx, dist, None, newcameramtx)

    # Side-by-side comparison (always saved)
    img_to_test_resized = cv2.resize(img_to_test, (dst.shape[1], dst.shape[0]))
    comparison_img = np.hstack((img_to_test_resized, dst))

    cv2.putText(comparison_img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(comparison_img, "Undistorted", (dst.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    preview_path = os.path.join(OUTPUT_DIR, 'undistort_preview.png')
    cv2.imwrite(preview_path, comparison_img)
    print(f"💾 Saved undistortion preview to: {preview_path}")

    # Show if possible
    _safe_imshow('Original vs. Undistorted Image', comparison_img, delay=0)
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == '__main__':
    main()

