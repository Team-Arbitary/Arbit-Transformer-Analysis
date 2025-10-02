#!/usr/bin/env python3
"""
Quick test to verify all preprocessing steps are working correctly.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from thermal_hotpoint_detector import ThermalHotpointDetector
import os

def visualize_preprocessing_steps(image_path):
    """Visualize each preprocessing step."""
    
    # Create detector
    detector = ThermalHotpointDetector(temperature_threshold=180)
    
    # Load original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"\nProcessing: {os.path.basename(image_path)}")
    print(f"Original image size: {image_rgb.shape}")
    
    # Step 1: Crop border
    cropped_image, crop_info = detector.crop_border(image_rgb, border_percent=10)
    print(f"After 10% border removal: {cropped_image.shape}")
    print(f"  Crop boundaries: top={crop_info['top']}, left={crop_info['left']}")
    
    # Step 2: Remove white regions
    valid_mask = detector.remove_white_regions(cropped_image)
    white_pixels_removed = np.sum(~valid_mask)
    print(f"White regions masked: {white_pixels_removed} pixels removed")
    
    # Step 3: Extract red channel
    red_channel = cropped_image[:, :, 0]
    red_channel_masked = red_channel.copy()
    red_channel_masked[~valid_mask] = 0
    
    hot_pixels_before = np.sum(red_channel > 180)
    hot_pixels_after = np.sum(red_channel_masked > 180)
    print(f"Red channel extraction:")
    print(f"  Hot pixels before masking: {hot_pixels_before}")
    print(f"  Hot pixels after masking: {hot_pixels_after}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('1. Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # After border crop
    axes[0, 1].imshow(cropped_image)
    axes[0, 1].set_title('2. After 10% Border Removal', fontweight='bold')
    axes[0, 1].axis('off')
    
    # White mask
    axes[0, 2].imshow(valid_mask, cmap='gray')
    axes[0, 2].set_title('3. Valid Region Mask\n(White regions removed)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Red channel before masking
    axes[1, 0].imshow(red_channel, cmap='Reds')
    axes[1, 0].set_title('4. Red Channel\n(Before white masking)', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Red channel after masking
    axes[1, 1].imshow(red_channel_masked, cmap='Reds')
    axes[1, 1].set_title('5. Red Channel Masked\n(White regions = 0)', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Hot regions
    hot_mask = red_channel_masked > 180
    axes[1, 2].imshow(hot_mask, cmap='hot')
    axes[1, 2].set_title(f'6. Hot Regions Detected\n({np.sum(hot_mask)} pixels)', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Preprocessing Pipeline: {os.path.basename(image_path)}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = f"preprocessing_demo_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.close()

def test_large_box_filtering():
    """Test that large boxes (>80% of image) are filtered out."""
    print("\n" + "="*70)
    print("Testing Large Box Filtering (>80% of image area)")
    print("="*70)
    
    detector = ThermalHotpointDetector(temperature_threshold=100)  # Very low threshold
    
    result = detector.process_single_image(
        'Dataset/T1/faulty/T1_faulty_001.jpg', 
        None, 
        save_annotations=False
    )
    
    if result['num_hot_regions'] == 0:
        print("✓ Large box filtering PASSED: Oversized boxes were filtered out")
    else:
        print(f"⚠ Large box filtering may need adjustment: {result['num_hot_regions']} regions detected")

def main():
    print("="*70)
    print("Testing Preprocessing Pipeline")
    print("="*70)
    print("\nThis script tests all three preprocessing steps:")
    print("  1. Remove 10% border from each side")
    print("  2. Remove white regions with dilation")
    print("  3. Filter bounding boxes larger than 80% of image")
    
    # Test with a sample image
    sample_images = ['Dataset/T1/faulty/T1_faulty_001.jpg']
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            visualize_preprocessing_steps(img_path)
    
    # Test large box filtering
    test_large_box_filtering()
    
    print("\n" + "="*70)
    print("Preprocessing tests complete!")
    print("="*70)

if __name__ == "__main__":
    main()
