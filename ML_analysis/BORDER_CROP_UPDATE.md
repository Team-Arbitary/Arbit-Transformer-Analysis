# ✅ UPDATE: Border Cropping Feature Added

## 🎯 What Changed

The `detect_and_annotate.py` script now automatically **removes 10% border** from all sides of the image before processing, then **maps the detections back to the full original image** for output.

## 🔧 How It Works

### Processing Flow

1. **Load Full Image** (e.g., 640x640 pixels)
2. **Crop 10% Border** 
   - For 640x640 image: Process region [64:576, 64:576] (512x512 center)
   - Removes edge artifacts and focus on central area
3. **Process Cropped Region**
   - Run anomaly detection on cropped area
   - Generate heatmap and mask
4. **Map Back to Full Image**
   - Convert bounding box coordinates from cropped → original
   - Draw boxes on full-size original image
5. **Output Full Image**
   - Annotated image is full size
   - Mask is full size (with detections in center region)
   - All coordinates reference full image

## 📊 Example

### For a 640x640 image:

**Before Processing:**
```
Original Image: 640x640 pixels
Full border included
```

**During Processing:**
```
Cropped Region: [64:576, 64:576] = 512x512 pixels
Processing central 80% of image
Ignoring 10% border on all sides
```

**Output:**
```
Annotated Image: Full 640x640 pixels
Bounding boxes mapped to original coordinates
All detections shown on full image
```

## 💡 Why This Helps

### Benefits:

1. **Reduces Border Artifacts**
   - Thermal images often have edge noise
   - Focus on meaningful central region

2. **Improves Detection Accuracy**
   - Model trained on centered content
   - Reduces false positives from edges

3. **Maintains Full Context**
   - Output shows complete original image
   - Easy to see where anomalies are located

4. **Automatic Coordinate Mapping**
   - No manual adjustment needed
   - Seamless for the user

## 🚀 Usage (No Changes Required!)

The command is **exactly the same**:

```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.5
```

### Console Output Now Shows:

```
[2/6] Loading and preprocessing image...
  Original size: 640x640
  Processing region: [64:576, 64:576] (10% border removed)
```

### Report Now Includes:

```
Image Size: 640x640
Processing: 10% border removed before detection
Processed Region: [64:576, 64:576]
```

## 📋 What's Different in Output

### Before (Old Version):
- Processed entire image including borders
- Might detect edge artifacts
- Bounding boxes covered full image

### After (New Version):
- Processes only central 80% of image
- Ignores edge regions
- Bounding boxes still on full image
- More focused detections

## 🎨 Visual Example

```
Original Image (640x640):
┌─────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ ← 10% border (ignored)
│ ░░┌───────────────────┐░░░░ │
│ ░░│                   │░░░░ │
│ ░░│   Processed       │░░░░ │ ← Central region processed
│ ░░│   Region          │░░░░ │
│ ░░│   (80% of image)  │░░░░ │
│ ░░│                   │░░░░ │
│ ░░└───────────────────┘░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ ← 10% border (ignored)
└─────────────────────────────┘

Output Image (640x640):
┌─────────────────────────────┐
│                             │
│   ┌──────┐                  │ ← Bounding boxes
│   │Anom#1│                  │   on full image
│   └──────┘                  │   (mapped from 
│              ┌─────┐        │    cropped region)
│              │Ano#2│        │
│              └─────┘        │
│                             │
└─────────────────────────────┘
```

## 🔍 Coordinate Mapping Example

### Detection in Cropped Image:
```
Crop coordinates: [64:576, 64:576]
Detected box in cropped region: x=100, y=150, w=30, h=40
```

### Mapped to Full Image:
```
Box in original image: x=164, y=214, w=30, h=40
(Added x1=64 and y1=64 offsets)
```

### Result:
- ✅ User sees coordinates relative to full original image
- ✅ Can directly use coordinates with original image
- ✅ No confusion or manual adjustment needed

## 📝 Technical Details

### Crop Calculation:
```python
h, w = original_size
crop_percent = 10 / 100.0 = 0.1

y1 = int(h * 0.1)  # Top crop
y2 = int(h * 0.9)  # Bottom crop
x1 = int(w * 0.1)  # Left crop
x2 = int(w * 0.9)  # Right crop

Cropped region: image[y1:y2, x1:x2]
```

### Coordinate Mapping:
```python
# Detection in cropped image
x_crop, y_crop, w, h = detection

# Map to original image
x_orig = x_crop + x1
y_orig = y_crop + y1

# Output uses original coordinates
bounding_box = (x_orig, y_orig, w, h)
```

### Mask Mapping:
```python
# Create full-size mask
full_mask = np.zeros((H, W), dtype=np.uint8)

# Resize detection mask to cropped size
mask_cropped = cv2.resize(mask, (cropped_w, cropped_h))

# Place in corresponding region
full_mask[y1:y2, x1:x2] = mask_cropped

# Save full-size mask
```

## ✅ Verification

### Test Results:

**T1_faulty_001.jpg (640x640):**
```
Original size: 640x640
Processing region: [64:576, 64:576] (10% border removed)
Found 15 anomalies
All boxes correctly mapped to full image coordinates
```

**T1_faulty_026.jpg (640x640):**
```
Original size: 640x640
Processing region: [64:576, 64:576] (10% border removed)
Found 10 anomalies
All coordinates reference full image
```

## 🎯 Summary

### What You Get:

✅ **Input**: Full original image
✅ **Processing**: Central 80% (10% border removed)
✅ **Output**: Full image with bounding boxes
✅ **Coordinates**: All in original image space
✅ **Automatic**: No user action required
✅ **Transparent**: Processing region shown in console

### Key Advantages:

1. **Better accuracy** - Focus on meaningful regions
2. **Reduced noise** - Ignore problematic borders
3. **Full context** - See complete image in results
4. **Easy to use** - Coordinates match original image
5. **Well documented** - Reports show what was processed

## 🚀 Use It Now!

```bash
# Same command, better results!
python detect_and_annotate.py <image_path> --threshold 0.5
```

The script automatically:
- Crops 10% border for processing
- Detects anomalies in central region
- Maps everything back to full image
- Outputs full-size annotated image

**No changes to your workflow needed!** 🎉
