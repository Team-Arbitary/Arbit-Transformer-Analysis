# API Fix Summary

## Issue
The API was returning **0 anomalies** for all images, even though the `unified_thermal_analysis.py` script was working correctly and detecting anomalies.

## Root Cause
The API's `analyze_thermal_image()` function had several critical issues:

### 1. **ML Analysis Issues**

#### Issue A: Incorrect use of `generate_anomaly_mask()`
**Before (WRONG):**
```python
# Generate anomaly mask directly from tensors
anomaly_mask = generate_anomaly_mask(input_tensor, reconstruction, threshold=threshold)
```

**Problem:** `generate_anomaly_mask()` expects a numpy array (anomaly map), not tensors.

**After (CORRECT):**
```python
# Use model's get_anomaly_map method first
with torch.no_grad():
    anomaly_map, reconstructed = ML_MODEL.get_anomaly_map(image_tensor)

# Convert to numpy
anomaly_map_np = anomaly_map.cpu().squeeze().numpy()

# THEN generate mask
mask = generate_anomaly_mask(anomaly_map_np, threshold=threshold)
```

#### Issue B: Wrong parameters to `find_contours_and_draw_boxes()`
**Before (WRONG):**
```python
ml_boxes, annotated_ml = find_contours_and_draw_boxes(
    cv2.imread(image_path),  # Reload image
    anomaly_mask,
    min_area=min_area,
    max_area=max_area,
    max_annotations=max_annotations,
    blue_threshold=blue_threshold,
    crop_coords=crop_coords  # Missing original_size!
)
```

**After (CORRECT):**
```python
annotated_ml, ml_boxes = find_contours_and_draw_boxes(
    original_bgr,      # Use BGR image from preprocess
    mask,              # Binary mask
    original_size,     # Required parameter!
    crop_coords,       # Crop coordinates
    min_area=min_area,
    max_area=max_area,
    max_annotations=max_annotations,
    blue_threshold=blue_threshold
)
```

#### Issue C: Wrong return order
The function returns `(annotated_image, boxes)` but the code expected `(boxes, annotated_image)`.

### 2. **Data Format Issues**

#### Issue D: ML box format
**Before (WRONG):**
```python
x, y, w, h = box['x'], box['y'], box['width'], box['height']
```

**Problem:** ML boxes use `box['bbox']` which is a tuple `(x, y, w, h)`.

**After (CORRECT):**
```python
x, y, w, h = box['bbox']  # Correct format
score = box.get('score', 0.0) / 100.0  # Also normalize score
```

#### Issue E: Thermal box coordinates
**Before (WRONG):**
```python
x, y, w, h = bbox  # Direct tuple unpacking
```

**Problem:** Thermal boxes need crop offset adjustment.

**After (CORRECT):**
```python
crop_info = results["thermal_results"]["crop_info"]
x = bbox.x + crop_info['left']
y = bbox.y + crop_info['top']
w = bbox.width
h = bbox.height
```

## Fixed Files
- `api.py` - Fixed ML analysis flow and data format handling

## Test Results

### Before Fix
```json
{
    "total_anomalies": 0,
    "anomalies": []
}
```

### After Fix

#### T1 Faulty Image
```json
{
    "total_anomalies": 6,
    "severity_distribution": {
        "HIGH": 3,
        "MEDIUM": 3,
        "LOW": 0
    },
    "anomalies": [
        {
            "id": 1,
            "type": "ml_anomaly",
            "confidence": 0.793,
            "severity": 0.793
        },
        {
            "id": 2,
            "type": "ml_anomaly",
            "confidence": 0.71,
            "severity": 0.71
        },
        {
            "id": 3,
            "type": "ml_anomaly",
            "confidence": 0.697,
            "severity": 0.697
        },
        {
            "id": 4,
            "type": "thermal_hotspot",
            "confidence": 1.0,
            "severity": 0.8
        },
        {
            "id": 5,
            "type": "thermal_hotspot",
            "confidence": 1.0,
            "severity": 0.8
        },
        {
            "id": 6,
            "type": "thermal_hotspot",
            "confidence": 1.0,
            "severity": 0.8
        }
    ]
}
```

#### T3 Faulty Image
```json
{
    "total_anomalies": 4,
    "severity_distribution": {
        "HIGH": 3,
        "MEDIUM": 1,
        "LOW": 0
    }
}
```

## Key Takeaways

1. **Follow the Working Code Pattern**: The `unified_thermal_analysis.py` script had the correct implementation. The API should have mirrored its approach exactly.

2. **ML Model Flow**:
   - `preprocess_image()` → returns tensor [1, 3, 256, 256], original BGR, crop coords
   - `model.get_anomaly_map(tensor)` → returns anomaly_map tensor
   - Convert to numpy → `anomaly_map.cpu().squeeze().numpy()`
   - `generate_anomaly_mask(numpy_array, threshold)` → returns binary mask
   - `find_contours_and_draw_boxes(image, mask, size, coords, ...)` → returns (annotated, boxes)

3. **Box Formats**:
   - ML boxes: `box['bbox']` = tuple `(x, y, w, h)`, `box['score']` = percentage (0-100)
   - Thermal boxes: bbox object with `.x`, `.y`, `.width`, `.height`, `.confidence`, needs crop offset adjustment

4. **Always Check Function Signatures**: When integrating existing code, carefully check:
   - Parameter order
   - Return value order
   - Expected data types (numpy vs tensor, tuple vs dict)
   - Coordinate systems (original vs cropped)

## Status
✅ **FIXED** - API now detects both ML anomalies and thermal hotspots correctly!
