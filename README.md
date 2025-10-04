# Unified Thermal Transformer Analysis

A single tool that combines ML-based anomaly detection and thermal hotpoint detection into one annotated image output.

## Overview

This project analyzes thermal images using two complementary methods:

- ML analysis: an autoencoder-based anomaly detector (red boxes)
- Thermal analysis: hotspot detection using the red channel (yellow boxes)

The results from both methods are combined into a single annotated image.

## Quick start

### Prerequisites

Create and activate a Python virtual environment named `thermal_env` and install the project dependencies (see `requirements.txt`):

```powershell
.\thermal_env\Scripts\Activate.ps1
```

On Unix-like systems use:

```bash
source thermal_env/bin/activate
```

### Example usage

Run the unified analysis on a single image:

```powershell
.\thermal_env\Scripts\Activate.ps1; python unified_thermal_analysis.py \
    Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.5 \
    --min-area 200 \
    --max-area 5000 \
    --max-annotations 3 \
    --blue-threshold 30
```

## Output

The script produces three files in the `unified_results/` directory:

1. `*_combined_annotated.jpg` — annotated image containing ML (red) and thermal (yellow) detections
2. `*_unified_analysis.png` — a 6-panel visualization summarizing the analysis
3. `*_unified_report.txt` — a text report with detailed detection information

## Color coding

| Detection type | Box color | Label |
|---------------|-----------|-------|
| ML anomaly | red | ML-1, ML-2, ML-3 |
| Thermal hotspot | yellow | TH-1, TH-2, TH-3 |

## Parameters

### ML analysis options
- `--threshold <0-1>`: detection threshold (default: 0.5). Lower values produce more detections.
- `--min-area <pixels>`: minimum detection area (default: 200).
- `--max-area <pixels>`: maximum detection area (default: 5000).
- `--max-annotations <N>`: maximum annotations to show (default: 3).
- `--blue-threshold <percent>`: maximum allowed percentage of blue pixels in a detection (default: 30).

### Thermal analysis options
- `--thermal-threshold <0-255>`: temperature threshold on the red channel (default: 200).
- `--thermal-min-cluster <N>`: minimum cluster size for DBSCAN (default: 15).
- `--thermal-epsilon <N>`: DBSCAN epsilon value for clustering (default: 20).

### Output
- `--output-dir <path>`: output directory (default: `unified_results`).

## Project structure

```
Arbit-Transformer-Analysis/
├── unified_thermal_analysis.py    # main analysis script
├── requirements.txt               # Python dependencies
├── thermal_env/                   # suggested virtual environment
├── Dataset/                       # thermal image datasets
│   ├── T1/faulty/
│   ├── T2/faulty/
│   └── ...
├── ML_analysis/                   # ML anomaly detection module
│   ├── detect_and_annotate.py     # ML detection functions
│   ├── model.py                   # autoencoder model
│   ├── train.py                   # model training script
│   └── models/                    # trained models
│       └── best_model.pth
├── heat_point_analysis/           # thermal hotspot detection module
│   └── thermal_hotpoint_detector.py
└── unified_results/               # output directory
```

## Examples

Run the main analysis on a single image:

```powershell
python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg
```

Show more detections by lowering the ML threshold and increasing the result limit:

```powershell
python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.3 \
    --max-annotations 5
```

Adjust both ML and thermal thresholds and save results to a custom folder:

```powershell
python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.5 \
    --thermal-threshold 180 \
    --output-dir my_results
```

## How it works

### ML analysis (autoencoder)
- Loads a trained autoencoder model
- Produces a reconstruction error heatmap
- Finds regions with high reconstruction error
- Filters detections by area and blue-channel content
- Returns the top N detections and draws red bounding boxes

### Thermal analysis (red channel)
- Uses the image red channel as a temperature indicator
- Thresholds hot pixels and clusters them with DBSCAN
- Builds bounding boxes around clusters and filters out border/white noise
- Draws yellow bounding boxes for hotspots

### Unified output
- Combines both sets of detections
- Produces a 6-panel visualization and a single annotated image
- Saves a text report with detection details

## Tips

- To get more ML detections, lower `--threshold` (for example, 0.3).
- To detect more thermal hotspots, lower `--thermal-threshold` (for example, 150).
- Increase `--max-annotations` to show more results (for example, 5).
- Increase `--min-area` to reduce small/noisy detections (for example, 500).

## Troubleshooting

If you see import errors, make sure the virtual environment is activated and dependencies are installed.

If the ML model cannot be found, confirm that `ML_analysis/models/best_model.pth` exists.

If you get no detections, try relaxing the thresholds:

```powershell
python unified_thermal_analysis.py <image> \
    --threshold 0.3 \
    --thermal-threshold 150
```

## Requirements

Install dependencies from `requirements.txt` into your virtual environment. Typical requirements include:

- torch >= 2.0.0
- torchvision >= 0.15.0
- opencv-python >= 4.8.0
- opencv-contrib-python >= 4.8.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scikit-learn >= 1.3.0
- albumentations >= 1.3.0

## Additional documentation

See `QUICKSTART.md` for a short setup guide and `ML_analysis/README.md` for details on the ML components.

---

Main command example:

```powershell
.\thermal_env\Scripts\Activate.ps1; python unified_thermal_analysis.py \
    Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.5 --min-area 200 --max-area 5000 \
    --max-annotations 3 --blue-threshold 30
```

Main output: `unified_results/*_combined_annotated.jpg`
