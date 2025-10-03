# 🎯 Quick Reference - Anomaly Detection Script

## One-Line Command
```bash
python detect_and_annotate.py <image_path> --threshold 0.5
```

## Common Use Cases

### 1. Basic Detection (Default Settings)
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg
```

### 2. More Sensitive (Catch Small Anomalies)
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.3 --min-area 50
```

### 3. Less Sensitive (Only Major Defects)
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.7 --min-area 300
```

### 4. Custom Output Location
```bash
python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --save-dir my_results
```

## Output Files (4 Files Per Image)

| File | Description |
|------|-------------|
| `*_annotated.jpg` | Image with red bounding boxes and labels |
| `*_mask.png` | Binary segmentation mask (white=anomaly) |
| `*_visualization.png` | Complete analysis dashboard |
| `*_report.txt` | Detailed text report |

## Parameters Cheat Sheet

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | Required | - | Path to thermal image |
| `--threshold` | 0.0-1.0 | 0.5 | Detection sensitivity |
| `--min-area` | Integer | 100 | Min anomaly size (pixels²) |
| `--model` | Path | models/best_model.pth | Model file |
| `--save-dir` | Path | annotated_results | Output folder |

## Threshold Guide

| Threshold | Sensitivity | Best For |
|-----------|-------------|----------|
| 0.3 | High | Finding all possible anomalies |
| 0.5 | Medium | Balanced detection (default) |
| 0.7 | Low | Only confident detections |

## Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| No anomalies found | Lower `--threshold` to 0.3 or 0.4 |
| Too many false positives | Raise `--threshold` to 0.6 or 0.7 |
| Small noise detected | Increase `--min-area` to 200+ |
| Missing small defects | Decrease `--min-area` to 50 |

## Example Output

```
Anomaly #1:
  Location: (x=540, y=408)
  Size: 18x77 pixels
  Area: 1120.0 pixels²
  Anomaly Score: 87.37%
```

## Batch Process All Images
```bash
# Process all T1 faulty images
for img in Dataset/T1/faulty/*.jpg; do
    python detect_and_annotate.py "$img" --threshold 0.5
done
```

## Full Command with All Options
```bash
python detect_and_annotate.py \
    Dataset/T1/faulty/T1_faulty_001.jpg \
    --model models/best_model.pth \
    --threshold 0.5 \
    --min-area 100 \
    --save-dir annotated_results
```

---

📖 **See DETECTION_GUIDE.md for complete documentation**
