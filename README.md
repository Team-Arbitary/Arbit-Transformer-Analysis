# Unified Thermal Transformer Analysis

A unified system that combines ML-based anomaly detection and thermal hotpoint detection into a single annotated image output.

## 🎯 Overview

This system analyzes thermal images using two complementary methods:
- **ML Analysis**: AutoEncoder-based anomaly detection (RED boxes)
- **Thermal Analysis**: Red channel-based hotpoint detection (YELLOW boxes)

Both analyses are combined into a single annotated output image.

## 🚀 Quick Start

### Prerequisites

Ensure `thermal_env` virtual environment is set up with all dependencies:
```bash
source thermal_env/bin/activate
```

### Usage

```bash
source thermal_env/bin/activate && python unified_thermal_analysis.py \
    Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.5 \
    --min-area 200 \
    --max-area 5000 \
    --max-annotations 3 \
    --blue-threshold 30
```

## 📊 Output

The script generates **3 files** in the `unified_results/` directory:

1. **`*_combined_annotated.jpg`** ⭐ - Main output with both ML (RED) and thermal (YELLOW) annotations
2. **`*_unified_analysis.png`** - 6-panel comprehensive visualization
3. **`*_unified_report.txt`** - Detailed text report with all detections

## 🎨 Color Coding

| Detection Type | Box Color | Label |
|---------------|-----------|-------|
| **ML Anomaly** | 🔴 RED | ML-1, ML-2, ML-3 |
| **Thermal Hotspot** | 🟡 YELLOW | TH-1, TH-2, TH-3 |

## ⚙️ Parameters

### ML Analysis Parameters
- `--threshold <0-1>` - Detection threshold (default: 0.5, lower = more detections)
- `--min-area <pixels>` - Minimum detection area (default: 200)
- `--max-area <pixels>` - Maximum detection area (default: 5000)
- `--max-annotations <N>` - Maximum annotations shown (default: 3)
- `--blue-threshold <percent>` - Maximum blue pixels percentage (default: 30)

### Thermal Analysis Parameters
- `--thermal-threshold <0-255>` - Temperature threshold (default: 200)
- `--thermal-min-cluster <N>` - Minimum cluster size (default: 15)
- `--thermal-epsilon <N>` - Cluster epsilon for DBSCAN (default: 20)

### Output
- `--output-dir <path>` - Output directory (default: unified_results)

## 📁 Project Structure

```
Arbit-Transformer-Analysis/
├── unified_thermal_analysis.py    # Main analysis script
├── requirements.txt                # Python dependencies
├── thermal_env/                    # Virtual environment
├── Dataset/                        # Thermal image datasets
│   ├── T1/faulty/
│   ├── T2/faulty/
│   └── ...
├── ML_analysis/                    # ML anomaly detection module
│   ├── detect_and_annotate.py     # ML detection functions
│   ├── model.py                    # AutoEncoder model
│   ├── train.py                    # Model training script
│   └── models/                     # Trained models
│       └── best_model.pth
├── heat_point_analysis/            # Thermal hotpoint detection module
│   └── thermal_hotpoint_detector.py
└── unified_results/                # Output directory
```

## 🔧 Examples

### Basic Usage
```bash
python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg
```

### Custom Threshold
```bash
python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.3 \
    --max-annotations 5
```

### Adjust Both ML and Thermal
```bash
python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.5 \
    --thermal-threshold 180 \
    --output-dir my_results
```

## 📊 How It Works

### 1. ML Analysis (AutoEncoder)
- Loads trained AutoEncoder model
- Generates anomaly reconstruction heatmap
- Detects regions with high reconstruction error
- Filters by area and blue content
- Shows top-N most confident detections
- **Draws RED bounding boxes**

### 2. Thermal Analysis (Red Channel)
- Extracts red channel as temperature indicator
- Applies temperature threshold
- Clusters hot pixels using DBSCAN
- Creates bounding boxes around clusters
- Filters out border and white regions
- **Draws YELLOW bounding boxes**

### 3. Unified Output
- Combines both analyses
- Creates comprehensive 6-panel visualization
- Generates single annotated image with both detection types
- Saves detailed text report

## 💡 Tips

- **More ML detections**: Lower `--threshold` (e.g., 0.3)
- **More thermal hotspots**: Lower `--thermal-threshold` (e.g., 150)
- **Show more results**: Increase `--max-annotations` (e.g., 5)
- **Less noise**: Increase `--min-area` (e.g., 500)

## 🐛 Troubleshooting

### No module found errors
Ensure virtual environment is activated:
```bash
source thermal_env/bin/activate
```

### ML model not found
Ensure the trained model exists at `ML_analysis/models/best_model.pth`

### No detections
Try lowering thresholds:
```bash
python unified_thermal_analysis.py <image> \
    --threshold 0.3 \
    --thermal-threshold 150
```

## 📝 Requirements

All dependencies are in `requirements.txt` and should be installed in `thermal_env`:
- torch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.8.0
- opencv-contrib-python >= 4.8.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.3.0
- albumentations >= 1.3.0

## 📖 Additional Documentation

For more detailed information, see:
- `QUICKSTART.md` - Quick reference guide
- `ML_analysis/README.md` - ML module documentation

## ✅ System Status

- ✅ ML Analysis: Operational
- ✅ Thermal Analysis: Operational
- ✅ Combined Output: Operational
- ✅ Documentation: Complete

---

**Main Command:**
```bash
source thermal_env/bin/activate && python unified_thermal_analysis.py \
    Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.5 --min-area 200 --max-area 5000 \
    --max-annotations 3 --blue-threshold 30
```

**Main Output:** `unified_results/*_combined_annotated.jpg` ⭐
