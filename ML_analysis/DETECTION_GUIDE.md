# 🎯 Anomaly Detection with Bounding Box Annotation

## Single-Command Detection Script

This script provides **one-click anomaly detection** with **bounding boxes and labels** for thermal transformer images.

## 🚀 Quick Start

### Basic Usage
```bash
# Activate environment
source venv/bin/activate

# Run detection on an image
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg
```

### With Custom Parameters
```bash
# Adjust threshold (0-1, lower = more sensitive)
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.4

# Set minimum area to filter small detections
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.5 --min-area 200

# Custom output directory
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --save-dir custom_output
```

## 📊 What It Does

1. **Loads trained AutoEncoder model**
2. **Processes thermal image** through the network
3. **Generates anomaly heatmap** (reconstruction error)
4. **Creates binary segmentation mask**
5. **Finds contours** in the mask
6. **Draws bounding boxes** around anomalies
7. **Labels each anomaly** with ID and confidence score
8. **Saves multiple outputs** (annotated image, mask, report, visualization)

## 🎨 Output Files

For each input image, the script generates:

### 1. Annotated Image (`*_annotated.jpg`)
- Original image with red bounding boxes
- Each anomaly labeled with:
  - Anomaly ID number
  - Confidence score (percentage)
- Blue contour outlines

### 2. Binary Mask (`*_mask.png`)
- White pixels = anomalies
- Black pixels = normal regions
- Resized to original image dimensions

### 3. Visualization (`*_visualization.png`)
- **Top row:**
  - Original image
  - Anomaly heatmap (color-coded)
  - Binary mask
- **Bottom row:**
  - Large annotated image with all bounding boxes
  - Side panel with detection statistics

### 4. Detection Report (`*_report.txt`)
- Detailed text report with:
  - Image information
  - Reconstruction error
  - Total anomalies found
  - Per-anomaly details (location, size, score)

## ⚙️ Parameters

### Required
- `image_path` - Path to input thermal image

### Optional
- `--model` - Path to trained model (default: `models/best_model.pth`)
- `--threshold` - Anomaly detection threshold, 0-1 (default: `0.5`)
  - Lower = more sensitive (more detections)
  - Higher = more strict (fewer detections)
- `--min-area` - Minimum anomaly area in pixels (default: `100`)
  - Filters out small noise detections
- `--save-dir` - Output directory (default: `annotated_results`)

## 📈 Understanding the Output

### Bounding Box Labels

Each detected anomaly shows:
```
Anomaly #1
85.5%
```
- **#1**: Unique ID for this detection
- **85.5%**: Percentage of pixels inside the box that are anomalous

### Anomaly Score Interpretation

- **90-100%**: Very strong anomaly signal
- **70-90%**: Strong anomaly
- **50-70%**: Moderate anomaly
- **30-50%**: Weak anomaly (might be noise)

### Console Output

```
Anomaly #1:
  Location: (x=540, y=408)      ← Top-left corner of bounding box
  Size: 18x77 pixels            ← Width x Height
  Area: 1120.0 pixels²          ← Total area covered
  Anomaly Score: 87.37%         ← Confidence score
```

## 🔧 Tuning for Best Results

### If Too Many False Positives
```bash
# Increase threshold
python detect_and_annotate.py image.jpg --threshold 0.7

# Or increase minimum area
python detect_and_annotate.py image.jpg --min-area 300
```

### If Missing Anomalies
```bash
# Decrease threshold
python detect_and_annotate.py image.jpg --threshold 0.3

# Or decrease minimum area
python detect_and_annotate.py image.jpg --min-area 50
```

### Recommended Starting Points

**For small, localized anomalies:**
```bash
python detect_and_annotate.py image.jpg --threshold 0.4 --min-area 50
```

**For large, obvious defects:**
```bash
python detect_and_annotate.py image.jpg --threshold 0.6 --min-area 300
```

**Balanced (default):**
```bash
python detect_and_annotate.py image.jpg --threshold 0.5 --min-area 100
```

## 📝 Example Workflows

### Batch Processing Multiple Images
```bash
# Process all faulty images in T1
for img in Dataset/T1/faulty/*.jpg; do
    python detect_and_annotate.py "$img" --threshold 0.5
done
```

### Compare Different Thresholds
```bash
# Try multiple thresholds on same image
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.3 --save-dir results_th03
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.5 --save-dir results_th05
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.7 --save-dir results_th07
```

### Quick Preview (No Saving)
```bash
# Just display visualization without saving
# (Remove --save-dir or set to empty)
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.5
```

## 🎓 Understanding the Detection Process

### Step 1: AutoEncoder Reconstruction
```
Input Image → Encoder → Latent Space → Decoder → Reconstructed Image
```

### Step 2: Error Calculation
```
Reconstruction Error = |Original - Reconstructed|
```
High error = Anomaly (model doesn't recognize pattern)

### Step 3: Thresholding
```
Binary Mask = (Error > Threshold) ? White : Black
```

### Step 4: Contour Detection
```
Binary Mask → Find Contours → Filter by Area → Draw Boxes
```

### Step 5: Labeling
```
Each Contour → Bounding Box → Label with ID + Score
```

## 🐛 Troubleshooting

### Issue: No anomalies detected
**Solutions:**
- Lower the threshold: `--threshold 0.3`
- Check if model is trained: `models/best_model.pth` should exist
- Verify image is faulty (test on known faulty image)

### Issue: Too many false positives
**Solutions:**
- Raise the threshold: `--threshold 0.7`
- Increase min area: `--min-area 300`
- Train model with more normal images

### Issue: Entire image marked as anomaly
**Causes:**
- Model not well-trained
- Image very different from training data
- Threshold too low

**Solutions:**
- Retrain with more epochs
- Increase threshold
- Check if image type matches training data

### Issue: Bounding boxes too large
**Solutions:**
- Adjust morphological kernel size in code
- Increase threshold to get tighter masks
- Use `--min-area` to filter large detections

## 💡 Tips

1. **Start with default parameters** and adjust based on results
2. **Visualize multiple thresholds** to find optimal value
3. **Check the heatmap** to understand detection reasoning
4. **Use min-area to filter noise** without changing threshold
5. **Save reports** for documentation and analysis

## 📊 Output Directory Structure

```
annotated_results/
├── T1_faulty_001_annotated.jpg       ← Main result with boxes
├── T1_faulty_001_mask.png            ← Binary segmentation mask
├── T1_faulty_001_visualization.png   ← Complete analysis panel
└── T1_faulty_001_report.txt          ← Text report with details
```

## 🔬 Advanced Usage

### Programmatic Usage

You can also import and use the function in your own scripts:

```python
from detect_and_annotate import detect_and_annotate

# Run detection
annotated_img, boxes, details = detect_and_annotate(
    image_path='Dataset/T1/faulty/T1_faulty_001.jpg',
    threshold=0.5,
    min_area=100,
    save_dir='my_results'
)

# Access results
print(f"Found {details['num_anomalies']} anomalies")
print(f"Reconstruction error: {details['reconstruction_error']}")

# Iterate through detections
for box in boxes:
    print(f"Anomaly {box['id']}: {box['score']:.1f}% at ({box['bbox']})")
```

## 📚 Related Files

- `model.py` - AutoEncoder architecture
- `train.py` - Model training script
- `test.py` - Batch testing script
- `README.md` - Complete project documentation

## 🎯 Summary

This script provides a **complete, production-ready solution** for:
- ✅ Detecting anomalies in thermal images
- ✅ Drawing bounding boxes around defects
- ✅ Labeling each anomaly with confidence
- ✅ Generating comprehensive reports
- ✅ Saving multiple visualization formats

**One command. Complete results. Ready to use!** 🚀
