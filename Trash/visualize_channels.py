#!/usr/bin/env python3
"""
Visualization script to show channel analysis of thermal images.
This helps understand why we analyze only the red channel.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def visualize_channels(image_path, output_dir="channel_analysis"):
    """
    Visualize the different color channels of a thermal image.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}")
        return
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract channels
    red_channel = image_rgb[:, :, 0]
    green_channel = image_rgb[:, :, 1]
    blue_channel = image_rgb[:, :, 2]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Thermal Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Red channel
    axes[0, 1].imshow(red_channel, cmap='Reds')
    axes[0, 1].set_title('RED Channel\n(Thermal Danger)', fontsize=12, fontweight='bold', color='red')
    axes[0, 1].axis('off')
    
    # Green channel
    axes[0, 2].imshow(green_channel, cmap='Greens')
    axes[0, 2].set_title('GREEN Channel\n(Mid-level)', fontsize=12, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # Blue channel
    axes[1, 0].imshow(blue_channel, cmap='Blues')
    axes[1, 0].set_title('BLUE Channel\n(Cool/Safe)', fontsize=12, fontweight='bold', color='blue')
    axes[1, 0].axis('off')
    
    # Histogram comparison
    axes[1, 1].hist(red_channel.ravel(), bins=256, color='red', alpha=0.6, label='Red')
    axes[1, 1].hist(green_channel.ravel(), bins=256, color='green', alpha=0.6, label='Green')
    axes[1, 1].hist(blue_channel.ravel(), bins=256, color='blue', alpha=0.6, label='Blue')
    axes[1, 1].set_title('Channel Intensity Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Red channel with threshold visualization
    threshold = 180
    hot_mask = red_channel > threshold
    red_with_threshold = red_channel.copy()
    axes[1, 2].imshow(red_channel, cmap='Reds')
    axes[1, 2].contour(hot_mask, colors='yellow', linewidths=2, levels=[0.5])
    axes[1, 2].set_title(f'RED Channel Hot Regions\n(Threshold: {threshold})', 
                         fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Thermal Image Channel Analysis\n{os.path.basename(image_path)}',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_channel_analysis.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved channel analysis to: {output_path}")
    
    plt.close()
    
    # Print statistics
    print(f"\nChannel Statistics for {os.path.basename(image_path)}:")
    print(f"  Red   - Mean: {red_channel.mean():.1f}, Max: {red_channel.max()}, Pixels > {threshold}: {(red_channel > threshold).sum()}")
    print(f"  Green - Mean: {green_channel.mean():.1f}, Max: {green_channel.max()}, Pixels > {threshold}: {(green_channel > threshold).sum()}")
    print(f"  Blue  - Mean: {blue_channel.mean():.1f}, Max: {blue_channel.max()}, Pixels > {threshold}: {(blue_channel > threshold).sum()}")
    print()

def main():
    # Get sample images
    dataset_path = "Dataset"
    sample_images = glob.glob(os.path.join(dataset_path, "T1/faulty/*.jpg"))[:3]
    
    if not sample_images:
        print("No sample images found!")
        return
    
    print("="*70)
    print("Thermal Image Channel Analysis")
    print("="*70)
    print("\nThis script shows why we analyze ONLY the RED channel:")
    print("  • RED pixels   = High temperature (thermal danger)")
    print("  • YELLOW pixels = Medium temperature (caution)")
    print("  • BLUE pixels  = Low temperature (safe)")
    print("\nBy extracting only the red channel, we focus on actual thermal issues.")
    print("="*70)
    print()
    
    for image_path in sample_images:
        visualize_channels(image_path)

if __name__ == "__main__":
    main()
