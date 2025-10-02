"""
Feature Matching and Image Transformation
==========================================
This script aligns faulty thermal images to normal reference images using 
feature matching and homography transformation, then saves the aligned images 
with the same filename as the normal reference images.

Features:
- 10% border cropping before feature detection
- SIFT feature detection for high accuracy
- RANSAC-based homography estimation
- Automatic batch processing for all transformer types
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path


def crop_border(image, border_percent=10):
    """
    Crop border_percent from all sides of the image.
    
    Args:
        image: Input image
        border_percent: Percentage of border to remove (default: 10%)
        
    Returns:
        cropped_image, crop_info (dict with original dimensions and crop amounts)
    """
    height, width = image.shape[:2]
    
    # Calculate crop amounts
    crop_h = int(height * border_percent / 100)
    crop_w = int(width * border_percent / 100)
    
    # Crop the image
    cropped = image[crop_h:height-crop_h, crop_w:width-crop_w]
    
    crop_info = {
        'original_height': height,
        'original_width': width,
        'crop_top': crop_h,
        'crop_bottom': crop_h,
        'crop_left': crop_w,
        'crop_right': crop_w,
        'cropped_height': cropped.shape[0],
        'cropped_width': cropped.shape[1]
    }
    
    return cropped, crop_info


def detect_and_match_features(img1, img2, method='sift', max_features=5000, crop_border_percent=10):
    """
    Detect features in two images and match them with improved accuracy.
    Crops borders before feature detection to avoid edge artifacts.
    
    Args:
        img1: First image (reference/normal)
        img2: Second image (to be aligned/faulty)
        method: Feature detection method ('sift', 'orb', or 'akaze')
        max_features: Maximum number of features to detect
        crop_border_percent: Percentage of border to remove before feature detection
        
    Returns:
        keypoints1, keypoints2, good_matches, crop_info1, crop_info2
    """
    # Crop borders from both images
    img1_cropped, crop_info1 = crop_border(img1, crop_border_percent)
    img2_cropped, crop_info2 = crop_border(img2, crop_border_percent)
    
    print(f"  Original size: {img1.shape[:2]} -> Cropped: {img1_cropped.shape[:2]}")
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1_cropped, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_cropped, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve feature detection
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)
    
    # Choose feature detector - SIFT is most accurate for thermal images
    if method.lower() == 'sift':
        detector = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.03, edgeThreshold=15)
    elif method.lower() == 'akaze':
        detector = cv2.AKAZE_create()
    else:  # ORB
        detector = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    
    # Detect keypoints and compute descriptors on cropped images
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    # Adjust keypoint coordinates back to original image space
    for kp in kp1:
        kp.pt = (kp.pt[0] + crop_info1['crop_left'], kp.pt[1] + crop_info1['crop_top'])
    
    for kp in kp2:
        kp.pt = (kp.pt[0] + crop_info2['crop_left'], kp.pt[1] + crop_info2['crop_top'])
    
    print(f"  Detected {len(kp1)} features in reference image")
    print(f"  Detected {len(kp2)} features in faulty image")
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print("  ✗ Not enough features detected!")
        return kp1, kp2, [], crop_info1, crop_info2
    
    # Match features with improved parameters
    if method.lower() == 'sift':
        # Use FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = matcher.knnMatch(des1, des2, k=2)
        except:
            print("  ✗ Matching failed!")
            return kp1, kp2, [], crop_info1, crop_info2
        
        # Apply Lowe's ratio test with stricter threshold
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.65 * n.distance:
                    good_matches.append(m)
    
    elif method.lower() == 'akaze':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
    
    else:  # ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
    
    # Sort by distance
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    print(f"  Found {len(good_matches)} good matches after ratio test")
    
    return kp1, kp2, good_matches, crop_info1, crop_info2


def align_images(img_reference, img_to_align, kp1, kp2, matches, min_match_count=20):
    """
    Align img_to_align to img_reference using robust homography transformation.
    
    Args:
        img_reference: Reference image (normal)
        img_to_align: Image to be aligned (faulty)
        kp1: Keypoints from reference image
        kp2: Keypoints from image to align
        matches: Good matches between keypoints
        min_match_count: Minimum number of matches required
        
    Returns:
        aligned_image, homography_matrix, match_status, inlier_mask
    """
    if len(matches) < min_match_count:
        print(f"  ✗ Not enough matches: {len(matches)} < {min_match_count}")
        return None, None, False, None
    
    # Extract matched keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography matrix with RANSAC
    homography, mask = cv2.findHomography(
        dst_pts, src_pts, cv2.RANSAC, 
        ransacReprojThreshold=3.0, 
        maxIters=5000, 
        confidence=0.995
    )
    
    if homography is None:
        print("  ✗ Could not find homography")
        return None, None, False, None
    
    # Check if homography is reasonable
    det = np.linalg.det(homography[:2, :2])
    if det < 0.1 or det > 10:
        print(f"  ✗ Homography appears unrealistic (det={det:.2f}), rejecting")
        return None, None, False, None
    
    # Warp the image to align with reference
    height, width = img_reference.shape[:2]
    aligned_img = cv2.warpPerspective(
        img_to_align, homography, (width, height),
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT
    )
    
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(matches)
    
    print(f"  ✓ Homography found: {inliers}/{len(matches)} inliers ({inlier_ratio*100:.1f}%)")
    print(f"  ✓ Determinant: {det:.3f}")
    
    if inlier_ratio < 0.3:
        print(f"  ⚠ Warning: Low inlier ratio ({inlier_ratio*100:.1f}%), alignment may be poor")
    
    return aligned_img, homography, True, mask


def process_transformer(transformer_type, dataset_path="Dataset", output_path="Dataset_Aligned", 
                       method='sift', max_features=5000, crop_border_percent=10):
    """
    Process all faulty images for a transformer type and save aligned versions.
    
    Args:
        transformer_type: 'T1', 'T2', etc.
        dataset_path: Path to dataset directory
        output_path: Path to output directory
        method: Feature detection method
        max_features: Maximum features to detect
        crop_border_percent: Percentage of border to crop
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'='*70}")
    print(f"Processing: {transformer_type}")
    print(f"{'='*70}")
    
    # Setup paths
    base_path = os.path.join(dataset_path, transformer_type)
    output_base = os.path.join(output_path, transformer_type)
    
    # Get normal and faulty images
    normal_pattern = os.path.join(base_path, "normal/*.*")
    faulty_pattern = os.path.join(base_path, "faulty/*.*")
    
    normal_images = sorted(glob.glob(normal_pattern))
    faulty_images = sorted(glob.glob(faulty_pattern))
    
    print(f"\nFound {len(normal_images)} normal image(s)")
    print(f"Found {len(faulty_images)} faulty image(s)")
    
    if len(normal_images) == 0:
        print(f"✗ No normal images found for {transformer_type}")
        return {'transformer': transformer_type, 'processed': 0, 'successful': 0, 'failed': 0}
    
    if len(faulty_images) == 0:
        print(f"✗ No faulty images found for {transformer_type}")
        return {'transformer': transformer_type, 'processed': 0, 'successful': 0, 'failed': 0}
    
    # Use first normal image as reference
    normal_path = normal_images[0]
    normal_filename = os.path.basename(normal_path)
    print(f"\nReference image: {normal_filename}")
    
    # Load reference image
    normal_img = cv2.imread(normal_path)
    if normal_img is None:
        print(f"✗ Could not load reference image: {normal_path}")
        return {'transformer': transformer_type, 'processed': 0, 'successful': 0, 'failed': 0}
    
    # Create output directory
    output_normal_dir = os.path.join(output_base, "normal")
    os.makedirs(output_normal_dir, exist_ok=True)
    
    # Save the reference normal image to output
    output_normal_path = os.path.join(output_normal_dir, normal_filename)
    cv2.imwrite(output_normal_path, normal_img)
    print(f"✓ Saved reference image to: {output_normal_path}")
    
    # Statistics
    stats = {
        'transformer': transformer_type,
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'details': []
    }
    
    # Process each faulty image
    print(f"\nProcessing {len(faulty_images)} faulty image(s)...")
    print("-" * 70)
    
    for idx, faulty_path in enumerate(faulty_images, 1):
        faulty_filename = os.path.basename(faulty_path)
        print(f"\n[{idx}/{len(faulty_images)}] {faulty_filename}")
        
        stats['processed'] += 1
        
        try:
            # Load faulty image
            faulty_img = cv2.imread(faulty_path)
            if faulty_img is None:
                raise ValueError(f"Could not load image: {faulty_path}")
            
            # Detect and match features
            kp1, kp2, matches, _, _ = detect_and_match_features(
                normal_img, faulty_img,
                method=method,
                max_features=max_features,
                crop_border_percent=crop_border_percent
            )
            
            # Align images
            aligned_img, homography, success, mask = align_images(
                normal_img, faulty_img, kp1, kp2, matches
            )
            
            if success and aligned_img is not None:
                # Save aligned image with same filename as normal image
                output_aligned_path = os.path.join(output_normal_dir, normal_filename)
                
                # If multiple faulty images, add index to filename
                if len(faulty_images) > 1:
                    name_parts = os.path.splitext(normal_filename)
                    output_aligned_path = os.path.join(
                        output_normal_dir, 
                        f"{name_parts[0]}_aligned_{idx}{name_parts[1]}"
                    )
                
                cv2.imwrite(output_aligned_path, aligned_img)
                print(f"  ✓ Saved aligned image to: {output_aligned_path}")
                
                stats['successful'] += 1
                stats['details'].append({
                    'faulty_file': faulty_filename,
                    'output_file': os.path.basename(output_aligned_path),
                    'num_matches': len(matches),
                    'status': 'success'
                })
            else:
                print(f"  ✗ Alignment failed")
                stats['failed'] += 1
                stats['details'].append({
                    'faulty_file': faulty_filename,
                    'status': 'failed',
                    'reason': 'alignment_failed'
                })
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            stats['failed'] += 1
            stats['details'].append({
                'faulty_file': faulty_filename,
                'status': 'error',
                'reason': str(e)
            })
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Summary for {transformer_type}:")
    print(f"  Total processed: {stats['processed']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {(stats['successful']/stats['processed']*100):.1f}%")
    print(f"{'='*70}")
    
    return stats


def main():
    """Main function to process all transformer types."""
    print("\n" + "="*70)
    print("Feature Matching and Image Alignment")
    print("="*70)
    print("\nConfiguration:")
    print("  - Method: SIFT")
    print("  - Max features: 5000")
    print("  - Border crop: 10%")
    print("  - Output directory: Dataset_Aligned/")
    
    # Dataset configuration
    dataset_path = "Dataset"
    output_path = "Dataset_Aligned"
    
    # Find all transformer types
    transformer_types = []
    if os.path.exists(dataset_path):
        transformer_types = sorted([d for d in os.listdir(dataset_path) 
                                   if os.path.isdir(os.path.join(dataset_path, d)) 
                                   and d.startswith('T')])
    
    if not transformer_types:
        print(f"\n✗ No transformer directories found in {dataset_path}")
        return
    
    print(f"\nFound transformer types: {', '.join(transformer_types)}")
    
    # Process each transformer type
    all_stats = []
    for transformer_type in transformer_types:
        stats = process_transformer(
            transformer_type,
            dataset_path=dataset_path,
            output_path=output_path,
            method='sift',
            max_features=5000,
            crop_border_percent=10
        )
        all_stats.append(stats)
    
    # Print overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    total_processed = sum(s['processed'] for s in all_stats)
    total_successful = sum(s['successful'] for s in all_stats)
    total_failed = sum(s['failed'] for s in all_stats)
    
    print(f"\nTransformers processed: {len(all_stats)}")
    print(f"Total images processed: {total_processed}")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    
    if total_processed > 0:
        print(f"Overall success rate: {(total_successful/total_processed*100):.1f}%")
    
    print(f"\n✓ All aligned images saved to: {output_path}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
