# ✅ UPDATE: Top Confident Annotations with Area Filtering

## 🎯 What's New

The `detect_and_annotate.py` script has been enhanced with three major improvements:

1. **Max Area Parameter** - Filter out anomalies that are too large
2. **Labels Only Display** - Show only text labels (no bounding boxes or contours)
3. **Top N Confident** - Display only the most confident detections (default: 3)

## 🚀 New Features

### 1. Maximum Area Filtering

**Parameter**: `--max-area`

Filter out anomalies larger than a specified size to avoid false positives from large regions.

```bash
# Only detect anomalies between 200 and 5000 pixels²
python detect_and_annotate.py image.jpg --min-area 200 --max-area 5000
```

**Use Case**: Exclude entire image sections or very large regions that might be lighting artifacts.

### 2. Labels Only (No Boxes or Contours)

The script now shows **clean labels only** without drawing:
- ❌ No red bounding boxes
- ❌ No blue contour outlines
- ✅ Only floating text labels with confidence scores

**Result**: Cleaner, less cluttered visualization focused on labeling anomalies.

### 3. Top N Most Confident Detections

**Parameter**: `--max-annotations` (default: 3)

Automatically ranks all detections by confidence score and shows only the top N.

```bash
# Show only the top 5 most confident anomalies
python detect_and_annotate.py image.jpg --max-annotations 5

# Show only the most confident anomaly
python detect_and_annotate.py image.jpg --max-annotations 1
```

**Ranking**: Sorted by anomaly score (percentage) from highest to lowest.

## 📊 How It Works

### Detection & Ranking Process

```
1. Detect all anomalies meeting min/max area criteria
2. Calculate confidence score for each (% of pixels anomalous)
3. Sort by confidence (highest to lowest)
4. Take top N (default: 3)
5. Assign IDs 1, 2, 3 to top detections
6. Display labels only (no boxes)
```

### Label Display

Labels are positioned:
- **Center-top** of each anomaly region
- **Semi-transparent red background**
- **White text** for visibility
- **Two lines**: Anomaly ID + Confidence %

Example label:
```
┌─────────────┐
│ Anomaly #1  │
│   81.5%     │
└─────────────┘
```

## 🎨 Visual Comparison

### Before (Old Version):
- All detected anomalies shown
- Red bounding boxes around each
- Blue contour outlines
- Can be cluttered with many detections

### After (New Version):
- Only top 3 (or specified N) shown
- Clean floating labels only
- No boxes or contours
- Focus on most confident anomalies

## 📝 Command Examples

### Example 1: Default (Top 3 Confident)
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.5
```

**Output**:
```
RESULTS: Showing top 3 anomalies (ranked by confidence)

Anomaly #1:
  Anomaly Score: 81.47%
  
Anomaly #2:
  Anomaly Score: 79.32%
  
Anomaly #3:
  Anomaly Score: 71.00%
```

### Example 2: With Area Limits
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.5 \
    --min-area 200 \
    --max-area 5000 \
    --max-annotations 3
```

**Filters**:
- Area between 200-5000 pixels²
- Shows top 3 by confidence

### Example 3: Show Only Best Detection
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_026.jpg \
    --threshold 0.6 \
    --max-annotations 1
```

**Output**: Single most confident anomaly

### Example 4: More Annotations
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_027.jpg \
    --threshold 0.5 \
    --max-annotations 5
```

**Output**: Top 5 confident anomalies

## 🔧 Parameters Summary

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--threshold` | float | 0.5 | Detection sensitivity (0-1) |
| `--min-area` | int | 100 | Minimum anomaly size (pixels²) |
| `--max-area` | int | None | Maximum anomaly size (pixels²) |
| `--max-annotations` | int | 3 | Number of top detections to show |

## 💡 Use Cases

### Use Case 1: Quality Control - Show Top Issues
```bash
# Show only the 3 most significant defects
python detect_and_annotate.py transformer_image.jpg --max-annotations 3
```

### Use Case 2: Filter Out Background
```bash
# Ignore very large regions (likely background)
python detect_and_annotate.py transformer_image.jpg --max-area 10000
```

### Use Case 3: Focus on Primary Anomaly
```bash
# Show only the most critical defect
python detect_and_annotate.py transformer_image.jpg --max-annotations 1
```

### Use Case 4: Medium-Sized Defects Only
```bash
# Find defects between 300-5000 pixels²
python detect_and_annotate.py transformer_image.jpg \
    --min-area 300 --max-area 5000 --max-annotations 3
```

## 📊 Test Results

### Test 1: T1_faulty_001.jpg
**Command**:
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.5 --min-area 200 --max-area 5000 --max-annotations 3
```

**Results**:
```
Top 3 Anomalies (by confidence):
  #1: 81.47% confidence (687 pixels²)
  #2: 79.32% confidence (1602 pixels²)
  #3: 71.00% confidence (250 pixels²)
```

### Test 2: T1_faulty_026.jpg
**Command**:
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_026.jpg \
    --threshold 0.55 --min-area 300 --max-area 10000 --max-annotations 3
```

**Results**:
```
Top 3 Anomalies (by confidence):
  #1: 78.63% confidence (327 pixels²)
  #2: 73.08% confidence (340 pixels²)
  #3: 72.27% confidence (1295 pixels²)
```

## 🎯 Benefits

### 1. Cleaner Visualization
- ✅ No cluttered boxes or contours
- ✅ Clear floating labels
- ✅ Easy to read and understand

### 2. Focused on What Matters
- ✅ Automatically ranks by confidence
- ✅ Shows only top N detections
- ✅ Highlights most critical issues

### 3. Flexible Filtering
- ✅ Filter by size (min/max area)
- ✅ Control number of annotations
- ✅ Adjustable confidence threshold

### 4. Better for Reporting
- ✅ Top 3 anomalies clearly identified
- ✅ Confidence scores visible
- ✅ Clean professional output

## 📝 Console Output Format

```
======================================================================
THERMAL TRANSFORMER ANOMALY DETECTION - TOP CONFIDENT ANOMALIES
======================================================================

Input Image: Dataset/T1/faulty/T1_faulty_001.jpg
Threshold: 0.5
Min Area: 200 pixels²
Max Area: 5000 pixels²
Max Annotations: 3 (top by confidence)
Device: cpu

[1/5] Loading model...
[2/6] Loading and preprocessing image...
  Original size: 640x640
  Processing region: [64:576, 64:576] (10% border removed)
[3/6] Generating anomaly map...
  Reconstruction error: 1.908225
[4/6] Generating segmentation mask...
[5/6] Detecting and annotating anomalies...
  (Mapping detections from cropped region to full image)
  (Showing top 3 most confident detections)

======================================================================
RESULTS: Showing top 3 anomalies (ranked by confidence)
======================================================================

Detailed Anomaly Information:
----------------------------------------------------------------------
  Anomaly #1:
    Location: (x=560, y=330)
    Size: 16x58 pixels
    Area: 686.5 pixels²
    Anomaly Score: 81.47%

  Anomaly #2:
    Location: (x=198, y=276)
    Size: 38x56 pixels
    Area: 1601.5 pixels²
    Anomaly Score: 79.32%

  Anomaly #3:
    Location: (x=310, y=368)
    Size: 20x20 pixels
    Area: 249.5 pixels²
    Anomaly Score: 71.00%
```

## 🔍 Technical Details

### Ranking Algorithm
```python
1. Detect all contours in mask
2. Filter by min_area and max_area
3. Calculate confidence score for each
4. Sort by score (descending)
5. Take top N (max_annotations)
6. Assign sequential IDs (1, 2, 3, ...)
```

### Confidence Score Calculation
```python
confidence = (anomalous_pixels / total_pixels_in_bbox) * 100
```

Higher score = More pixels in the bounding box are anomalous

### Label Positioning
```python
label_x = center_x - (label_width / 2)
label_y = top_y - 10  # 10 pixels above anomaly
```

Labels are automatically adjusted to stay within image bounds.

## 🎓 Best Practices

### 1. Start with Defaults
```bash
python detect_and_annotate.py image.jpg
```
This shows top 3 by confidence - good for most cases.

### 2. Adjust if Too Many/Few
```bash
# Too many small detections? Increase min-area
python detect_and_annotate.py image.jpg --min-area 500

# Too many large false positives? Add max-area
python detect_and_annotate.py image.jpg --max-area 5000

# Want more/fewer shown? Change max-annotations
python detect_and_annotate.py image.jpg --max-annotations 5
```

### 3. For Critical Applications
```bash
# Show only the single most confident anomaly
python detect_and_annotate.py image.jpg --max-annotations 1 --threshold 0.6
```

## ✅ Summary of Changes

| Feature | Before | After |
|---------|--------|-------|
| **Display** | All detections with boxes | Top N labels only |
| **Ranking** | No ranking | Sorted by confidence |
| **Max Area** | Not available | Configurable filter |
| **Clutter** | Can be cluttered | Clean, focused |
| **Annotations** | All anomalies | Top 3 (or N) only |
| **Boxes** | Red rectangles | No boxes |
| **Contours** | Blue outlines | No contours |
| **Labels** | With boxes | Floating labels only |

## 🚀 Quick Command Reference

```bash
# Default: Top 3 confident, labels only
python detect_and_annotate.py image.jpg

# Show top 5 confident
python detect_and_annotate.py image.jpg --max-annotations 5

# Filter by area (200-5000 pixels²)
python detect_and_annotate.py image.jpg --min-area 200 --max-area 5000

# Show only #1 most confident
python detect_and_annotate.py image.jpg --max-annotations 1

# Full custom configuration
python detect_and_annotate.py image.jpg \
    --threshold 0.55 \
    --min-area 300 \
    --max-area 10000 \
    --max-annotations 3
```

## 🎉 Result

You now have a **clean, focused anomaly detection system** that:

✅ Shows only the most important anomalies
✅ Ranks by confidence automatically
✅ Displays clean labels without boxes
✅ Filters by size (min/max area)
✅ Fully configurable parameters
✅ Professional, uncluttered output

**Perfect for quality control and reporting!** 🚀
