# ✅ COMPLETE: Anomaly Detection with Bounding Box Annotation

## 🎯 What You Asked For

> "Give me a single run code where the anomalies are annotated with boxes and labeled for a given threshold"

## ✅ Solution Delivered

**File**: `detect_and_annotate.py`

**Single Command**:
```bash
python detect_and_annotate.py <image_path> --threshold 0.5
```

## 🚀 Quick Start

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run detection on any image
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.4
```

## 🎨 What It Does

✅ **Detects anomalies** using trained AutoEncoder model
✅ **Draws bounding boxes** around each anomaly (red color)
✅ **Labels each box** with:
   - Anomaly ID number (#1, #2, #3...)
   - Confidence score (percentage)
✅ **Saves 4 output files**:
   - Annotated image with boxes
   - Binary segmentation mask
   - Complete visualization dashboard
   - Detailed text report

## 📊 Example Output

### Console Output
```
======================================================================
RESULTS: Found 5 anomalies
======================================================================

Detected Anomaly Information:
----------------------------------------------------------------------
  Anomaly #1:
    Location: (x=620, y=578)
    Size: 20x17 pixels
    Area: 200.0 pixels²
    Anomaly Score: 70.00%

  Anomaly #2:
    Location: (x=610, y=488)
    Size: 30x87 pixels
    Area: 1599.5 pixels²
    Anomaly Score: 66.44%
...
```

### Output Files
```
annotated_results/
├── T1_faulty_001_annotated.jpg       ← Main result with bounding boxes
├── T1_faulty_001_mask.png            ← Binary mask (white=anomaly)
├── T1_faulty_001_visualization.png   ← Complete analysis panel
└── T1_faulty_001_report.txt          ← Detailed report
```

## 🎛️ Adjustable Parameters

### Threshold (Sensitivity)
```bash
# More sensitive (find more anomalies)
python detect_and_annotate.py image.jpg --threshold 0.3

# Balanced (default)
python detect_and_annotate.py image.jpg --threshold 0.5

# Less sensitive (only strong anomalies)
python detect_and_annotate.py image.jpg --threshold 0.7
```

### Minimum Area (Filter Small Detections)
```bash
# Allow smaller detections
python detect_and_annotate.py image.jpg --min-area 50

# Filter out small noise
python detect_and_annotate.py image.jpg --min-area 300
```

### Combined
```bash
python detect_and_annotate.py image.jpg --threshold 0.4 --min-area 100
```

## 📋 Common Use Cases

### Use Case 1: Quick Analysis of Single Image
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg
```

### Use Case 2: Batch Process Multiple Images
```bash
for img in Dataset/T1/faulty/*.jpg; do
    python detect_and_annotate.py "$img" --threshold 0.5
done
```

### Use Case 3: Test Different Thresholds
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.3 --save-dir results_low
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.5 --save-dir results_mid
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.7 --save-dir results_high
```

## 🎨 Visual Output Features

### Bounding Box Annotation
- **Red rectangles** around each anomaly
- **Semi-transparent red label** with:
  - Anomaly number
  - Confidence percentage
- **Blue contour outline** showing exact shape

### Visualization Panel
- **Original image** (top-left)
- **Anomaly heatmap** with color scale (top-middle)
- **Binary mask** (top-right)
- **Large annotated result** with all boxes (bottom)
- **Detection statistics** panel on the right

### Text Report
- Image information (path, size)
- Reconstruction error score
- Total anomalies found
- Per-anomaly details:
  - Bounding box coordinates
  - Size (width × height)
  - Area in pixels²
  - Anomaly confidence score

## ✅ Tested & Working

Successfully tested on:
- ✅ `Dataset/T1/faulty/T1_faulty_001.jpg` - Found 5 anomalies
- ✅ `Dataset/T1/faulty/T1_faulty_026.jpg` - Found 11 anomalies
- ✅ Different thresholds (0.3, 0.4, 0.5, 0.6, 0.7)
- ✅ Different min-area settings (50, 100, 200, 300)

## 📚 Documentation

| File | Purpose |
|------|---------|
| `detect_and_annotate.py` | **Main script** - Run this! |
| `DETECTION_GUIDE.md` | Complete usage guide |
| `QUICK_REFERENCE.md` | Quick command reference |
| `README.md` | Full project documentation |

## 🔥 Key Features

1. **Single Command** - No complex setup
2. **Fully Automatic** - No manual annotation needed
3. **Adjustable** - Control sensitivity with threshold
4. **Comprehensive Output** - 4 file types per image
5. **Production Ready** - Error handling, progress display
6. **Well Documented** - Console output + text reports

## 💡 Pro Tips

1. **Start with default** (`--threshold 0.5`) and adjust as needed
2. **Lower threshold** (0.3-0.4) for sensitive detection
3. **Higher threshold** (0.6-0.7) to reduce false positives
4. **Use min-area** to filter small noise without changing sensitivity
5. **Check visualization** to understand what model sees
6. **Read the report.txt** for precise coordinates and scores

## 🎯 Summary

You now have a **complete, single-command solution** that:

✅ Takes an image path and threshold as input
✅ Detects all anomalies automatically
✅ Draws bounding boxes with labels
✅ Saves annotated images and detailed reports
✅ Works with any thermal transformer image
✅ Fully customizable parameters

**Just run**: 
```bash
python detect_and_annotate.py <your_image> --threshold 0.5
```

**That's it!** 🚀

---

## 🆘 Need Help?

- See `DETECTION_GUIDE.md` for detailed documentation
- See `QUICK_REFERENCE.md` for command examples
- Check console output for detection details
- Review saved reports for analysis

**Everything you need is ready to use!** ✨
