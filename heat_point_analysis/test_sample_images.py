#!/usr/bin/env python3
"""
Test script to process a few sample thermal images and visualize results.
"""

import os
import glob
from thermal_hotpoint_detector import ThermalHotpointDetector

def main():
    # Initialize detector with custom parameters
    # Adjust temperature_threshold based on your thermal images (0-255 range)
    # Higher threshold = only very hot regions detected
    # Lower threshold = more regions detected
    # NOTE: Analyzes RED channel only - Red = danger, Yellow/Blue = safe
    detector = ThermalHotpointDetector(
        temperature_threshold=230,  # Red channel threshold (adjust based on your images)
        min_cluster_size=20,        # Minimum pixels to form a hot region
        cluster_epsilon=25          # Maximum distance for clustering nearby pixels
    )
    
    # Define paths
    dataset_path = "Dataset"
    output_dir = "thermal_annotations_sample"
    
    # Get a few sample images from T1/faulty
    sample_images = glob.glob(os.path.join(dataset_path, "T1/faulty/*.jpg"))[:5]
    
    if not sample_images:
        print("No sample images found!")
        return
    
    print(f"Testing with {len(sample_images)} sample images...")
    print(f"Analyzing RED channel only (Red = danger, Yellow/Blue = safe)")
    print(f"Red channel threshold: {detector.temp_threshold}")
    print("="*60)
    
    # Process each sample image
    for idx, image_path in enumerate(sample_images, 1):
        try:
            print(f"\n[{idx}/{len(sample_images)}] {os.path.basename(image_path)}")
            results = detector.process_single_image(image_path, output_dir)
            
            if results['num_hot_regions'] > 0:
                print(f"  ✓ Detected {results['num_hot_regions']} hot region(s)")
                print(f"  ✓ Total hot pixels: {results['total_hot_pixels']}")
            else:
                print(f"  → No thermal issues detected (no pixels above threshold)")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"Sample processing complete!")
    print(f"Annotated images saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
