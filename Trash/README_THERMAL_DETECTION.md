# Thermal Hotpoint Detection for Transformer Analysis

This project provides automated detection and annotation of thermal hotspots in thermal images of transformers.

## Features

- **Red Channel Analysis**: Analyzes only the red channel from thermal images (Red = thermal danger, Yellow/Blue = safe)
- **Constant Threshold Detection**: Uses a fixed temperature threshold to identify truly hot regions (not just the hottest pixels)
- **Smart Clustering**: Groups nearby hot pixels into meaningful regions using DBSCAN algorithm
- **Non-overlapping Annotations**: Automatically merges overlapping bounding boxes
- **Visual Output**: Generates annotated images with rectangles around detected hot regions
- **JSON Annotations**: Saves detailed annotation data for further analysis
- **Batch Processing**: Process entire dataset or individual images

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv thermal_env
source thermal_env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Test with Sample Images

Process a few sample images to test the detector:

```bash
python test_sample_images.py
```

This will process 5 images from the T1/faulty directory and save annotated results to `thermal_annotations_sample/`.

### Process Entire Dataset

Process all faulty images in the dataset:

```bash
python thermal_hotpoint_detector.py
```

This will process all images in `Dataset/*/faulty/` directories and save results to `thermal_annotations/`.

### Adjust Detection Parameters

Edit the detector initialization in either script to adjust sensitivity:

```python
detector = ThermalHotpointDetector(
    temperature_threshold=180,  # Pixel intensity threshold (0-255)
                                # Lower = more sensitive
                                # Higher = only very hot regions
    min_cluster_size=20,        # Minimum pixels to form a region
                                # Higher = larger regions only
    cluster_epsilon=25          # Max distance to group pixels
                                # Lower = tighter grouping
)
```

### Process Individual Images

```python
from thermal_hotpoint_detector import ThermalHotpointDetector

detector = ThermalHotpointDetector(temperature_threshold=180)
results = detector.process_single_image(
    image_path="Dataset/T1/faulty/T1_faulty_001.jpg",
    output_dir="output"
)

print(f"Detected {results['num_hot_regions']} hot regions")
```

## Output Structure

For each processed image, the script generates:

1. **Annotated Image** (`*_annotated.png`):
   - Original thermal image with rectangles drawn around hot regions
   - Color-coded by confidence: Red (high), Yellow (medium), Orange (low)
   - Labels showing region number and confidence score

2. **Annotation JSON** (`*_annotations.json`):
   ```json
   {
     "image_path": "...",
     "num_hot_regions": 2,
     "bounding_boxes": [
       {
         "x": 100,
         "y": 150,
         "width": 50,
         "height": 60,
         "confidence": 0.85
       }
     ],
     "statistics": {
       "total_hot_pixels": 1234,
       "image_dimensions": [640, 480],
       "temperature_threshold": 180
     }
   }
   ```

3. **Processing Summary** (`processing_summary.json`):
   - Overall statistics for batch processing
   - Count of images with/without thermal issues

## How It Works

1. **Load Image**: Read thermal image and extract RED channel only (red = hot, yellow/blue = safe)
2. **Threshold Detection**: Apply constant threshold to red channel to identify pixels above temperature limit
3. **Morphological Filtering**: Clean up noise using opening/closing operations
4. **Clustering**: Group nearby hot pixels using DBSCAN algorithm
5. **Bounding Boxes**: Create rectangles around each cluster
6. **Merge Overlaps**: Combine overlapping boxes to avoid redundant annotations
7. **Visualize**: Draw annotations and save results

### Why Red Channel Only?

In thermal images:
- **Red pixels** = High temperature (thermal danger) ⚠️
- **Yellow/Blue pixels** = Normal/lower temperature (safe) ✅

By analyzing only the red channel, we focus on actual thermal issues and ignore safe regions.

## Adjusting for Your Images

If you're not getting good results, try adjusting:

- **Temperature Threshold** (most important):
  - Start with 180 and adjust based on results
  - If detecting too many false positives: increase threshold
  - If missing real hot spots: decrease threshold
  
- **Min Cluster Size**:
  - If detecting tiny irrelevant spots: increase this value
  - If missing small hot spots: decrease this value

- **Cluster Epsilon**:
  - If separate hot regions are being merged: decrease this value
  - If one hot region is split into multiple boxes: increase this value

## Notes

- Not all images may have thermal issues - this is expected!
- The constant threshold approach prevents false positives in normal images
- Overlapping annotations are automatically merged
- Images without hot regions will show "No thermal issues detected"

## Dependencies

- opencv-python >= 4.8.0
- numpy >= 1.21.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.5.0
