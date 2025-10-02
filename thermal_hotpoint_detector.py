#!/usr/bin/env python3
"""
Thermal Hotpoint Detector and Annotator

This script analyzes thermal images to detect hot regions and cluster them into 
rectangular annotations. It processes faulty images from the dataset and 
identifies areas with elevated temperatures.

Author: GitHub Copilot
Date: October 2, 2025
"""

import cv2
import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import namedtuple
import json

# Define a structure for bounding boxes
BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'width', 'height', 'confidence'])

class ThermalHotpointDetector:
    def __init__(self, temperature_threshold_percentile=85, min_cluster_size=10, cluster_epsilon=15):
        """
        Initialize the thermal hotpoint detector.
        
        Args:
            temperature_threshold_percentile (int): Percentile for temperature threshold (85 = top 15% hottest pixels)
            min_cluster_size (int): Minimum number of pixels to form a cluster
            cluster_epsilon (int): Maximum distance between samples in a cluster
        """
        self.temp_threshold_percentile = temperature_threshold_percentile
        self.min_cluster_size = min_cluster_size
        self.cluster_epsilon = cluster_epsilon
        
    def load_thermal_image(self, image_path):
        """
        Load a thermal image and convert to appropriate format.
        
        Args:
            image_path (str): Path to the thermal image
            
        Returns:
            tuple: (original_image, grayscale_image)
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB for matplotlib compatibility
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for temperature analysis
        # In thermal images, intensity often correlates with temperature
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image_rgb, gray
    
    def detect_hot_regions(self, thermal_image):
        """
        Detect hot regions in the thermal image using adaptive thresholding.
        
        Args:
            thermal_image (numpy.ndarray): Grayscale thermal image
            
        Returns:
            numpy.ndarray: Binary mask of hot regions
        """
        # Calculate dynamic threshold based on image statistics
        threshold_value = np.percentile(thermal_image, self.temp_threshold_percentile)
        
        # Create binary mask for hot regions
        hot_mask = thermal_image > threshold_value
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        hot_mask = cv2.morphologyEx(hot_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_CLOSE, kernel)
        
        return hot_mask.astype(bool)
    
    def cluster_hot_points(self, hot_mask):
        """
        Cluster hot points using DBSCAN algorithm.
        
        Args:
            hot_mask (numpy.ndarray): Binary mask of hot regions
            
        Returns:
            list: List of clustered point groups
        """
        # Find coordinates of hot pixels
        hot_points = np.column_stack(np.where(hot_mask))
        
        if len(hot_points) == 0:
            return []
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.cluster_epsilon, min_samples=self.min_cluster_size)
        cluster_labels = clustering.fit_predict(hot_points)
        
        # Group points by cluster
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            cluster_points = hot_points[cluster_labels == label]
            clusters.append(cluster_points)
        
        return clusters
    
    def create_bounding_boxes(self, clusters):
        """
        Create bounding boxes around clustered hot regions.
        
        Args:
            clusters (list): List of clustered point groups
            
        Returns:
            list: List of BoundingBox objects
        """
        bounding_boxes = []
        
        for i, cluster in enumerate(clusters):
            if len(cluster) < self.min_cluster_size:
                continue
                
            # Calculate bounding box coordinates
            min_row, min_col = np.min(cluster, axis=0)
            max_row, max_col = np.max(cluster, axis=0)
            
            # Add some padding around the bounding box
            padding = 5
            min_row = max(0, min_row - padding)
            min_col = max(0, min_col - padding)
            
            width = max_col - min_col + 2 * padding
            height = max_row - min_row + 2 * padding
            
            # Calculate confidence based on cluster density
            area = width * height
            density = len(cluster) / area if area > 0 else 0
            confidence = min(1.0, density / 0.1)  # Normalize density to confidence
            
            bbox = BoundingBox(
                x=min_col,
                y=min_row,
                width=width,
                height=height,
                confidence=confidence
            )
            bounding_boxes.append(bbox)
        
        return bounding_boxes
    
    def annotate_image(self, image, bounding_boxes, show_confidence=True):
        """
        Annotate the image with bounding boxes around hot regions.
        
        Args:
            image (numpy.ndarray): Original RGB image
            bounding_boxes (list): List of BoundingBox objects
            show_confidence (bool): Whether to display confidence scores
            
        Returns:
            matplotlib.figure.Figure: Annotated image figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('Thermal Image with Hot Region Detection', fontsize=14, fontweight='bold')
        
        # Draw bounding boxes
        for i, bbox in enumerate(bounding_boxes):
            # Color code by confidence: red for high confidence, yellow for low
            color = 'red' if bbox.confidence > 0.7 else 'yellow' if bbox.confidence > 0.4 else 'orange'
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (bbox.x, bbox.y), bbox.width, bbox.height,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect)
            
            # Add confidence label if requested
            if show_confidence:
                ax.text(
                    bbox.x, bbox.y - 5,
                    f'Hot Region {i+1}\nConf: {bbox.confidence:.2f}',
                    fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )
        
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def process_single_image(self, image_path, output_dir=None, save_annotations=True):
        """
        Process a single thermal image and detect hot regions.
        
        Args:
            image_path (str): Path to the thermal image
            output_dir (str): Directory to save annotated images
            save_annotations (bool): Whether to save annotation data
            
        Returns:
            dict: Processing results containing bounding boxes and statistics
        """
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Load image
        original_image, thermal_image = self.load_thermal_image(image_path)
        
        # Detect hot regions
        hot_mask = self.detect_hot_regions(thermal_image)
        
        # Cluster hot points
        clusters = self.cluster_hot_points(hot_mask)
        
        # Create bounding boxes
        bounding_boxes = self.create_bounding_boxes(clusters)
        
        # Create annotated visualization
        fig = self.annotate_image(original_image, bounding_boxes)
        
        # Save results if output directory is specified
        if output_dir and save_annotations:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save annotated image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_annotated.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            
            # Save annotation data
            annotation_data = {
                'image_path': image_path,
                'num_hot_regions': len(bounding_boxes),
                'bounding_boxes': [
                    {
                        'x': bbox.x,
                        'y': bbox.y,
                        'width': bbox.width,
                        'height': bbox.height,
                        'confidence': bbox.confidence
                    }
                    for bbox in bounding_boxes
                ],
                'statistics': {
                    'total_hot_pixels': int(np.sum(hot_mask)),
                    'image_dimensions': thermal_image.shape,
                    'temperature_threshold_percentile': self.temp_threshold_percentile
                }
            }
            
            json_path = os.path.join(output_dir, f"{base_name}_annotations.json")
            with open(json_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
        
        plt.show()
        
        # Return results
        results = {
            'bounding_boxes': bounding_boxes,
            'num_hot_regions': len(bounding_boxes),
            'total_hot_pixels': int(np.sum(hot_mask)),
            'clusters': clusters,
            'hot_mask': hot_mask
        }
        
        return results
    
    def process_dataset(self, dataset_path, output_dir="output_annotations"):
        """
        Process all faulty images in the dataset.
        
        Args:
            dataset_path (str): Path to the dataset directory
            output_dir (str): Directory to save all annotations
        """
        # Find all faulty image directories
        faulty_patterns = [
            os.path.join(dataset_path, "*/faulty/*.jpg"),
            os.path.join(dataset_path, "*/faulty/*.png")
        ]
        
        all_faulty_images = []
        for pattern in faulty_patterns:
            all_faulty_images.extend(glob.glob(pattern))
        
        if not all_faulty_images:
            print(f"No faulty images found in {dataset_path}")
            return
        
        print(f"Found {len(all_faulty_images)} faulty images to process")
        
        # Process each image
        all_results = []
        for image_path in all_faulty_images:
            try:
                results = self.process_single_image(image_path, output_dir)
                results['image_path'] = image_path
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        # Save summary statistics
        if output_dir:
            summary = {
                'total_images_processed': len(all_results),
                'total_hot_regions_detected': sum(r['num_hot_regions'] for r in all_results),
                'average_hot_regions_per_image': np.mean([r['num_hot_regions'] for r in all_results]) if all_results else 0,
                'images_with_hot_regions': len([r for r in all_results if r['num_hot_regions'] > 0])
            }
            
            summary_path = os.path.join(output_dir, "processing_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nProcessing complete! Results saved to: {output_dir}")
            print(f"Summary: {summary}")


def main():
    """
    Main function to demonstrate usage of the ThermalHotpointDetector.
    """
    # Initialize detector with custom parameters
    detector = ThermalHotpointDetector(
        temperature_threshold_percentile=80,  # Top 20% hottest pixels
        min_cluster_size=15,                  # Minimum 15 pixels per cluster
        cluster_epsilon=20                    # Maximum 20 pixel distance for clustering
    )
    
    # Define paths
    dataset_path = "Dataset"
    output_dir = "thermal_annotations"
    
    # Process the entire dataset
    print("Starting thermal hotpoint detection on faulty images...")
    detector.process_dataset(dataset_path, output_dir)
    
    # Example: Process a single image
    # single_image_path = "Dataset/T1/faulty/T1_faulty_001.jpg"
    # if os.path.exists(single_image_path):
    #     print(f"\nProcessing single image example: {single_image_path}")
    #     results = detector.process_single_image(single_image_path, output_dir)
    #     print(f"Detected {results['num_hot_regions']} hot regions")


if __name__ == "__main__":
    main()