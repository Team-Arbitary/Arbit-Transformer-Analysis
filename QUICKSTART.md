# 🚀 QUICK START GUIDE - Unified Thermal Analysis

## ⚡ One-Liner Commands

### Process Single Image (Easiest)
```bash
./run_unified_analysis.sh Dataset/T1/faulty/T1_faulty_001.jpg
```

### Process with Custom Parameters
```bash
source thermal_env/bin/activate && python unified_thermal_analysis.py \
    Dataset/T1/faulty/T1_faulty_001.jpg \
    --threshold 0.5 --min-area 200 --max-area 5000 \
    --max-annotations 3 --blue-threshold 30
```

### Batch Process Multiple Images
```bash
./batch_process.sh Dataset/T1/faulty 5
```

---

## 📊 Output Files

After running, check `unified_results/` folder:

| File | Purpose | Size |
|------|---------|------|
| `*_combined_annotated.jpg` | ⭐ **Main output** (RED + YELLOW boxes) | ~80 KB |
| `*_unified_analysis.png` | Full 6-panel visualization | ~2.4 MB |
| `*_unified_report.txt` | Detailed text report | ~1 KB |

---

## 🎨 Color Code

| Color | Detection Type | Label |
|-------|---------------|-------|
| 🔴 **RED** | ML Anomaly (AutoEncoder) | ML-1, ML-2, ML-3 |
| 🟡 **YELLOW** | Thermal Hotspot (Red Channel) | TH-1, TH-2, TH-3 |

---

## ⚙️ Key Parameters

```bash
--threshold 0.5          # ML detection sensitivity (lower = more detections)
--min-area 200           # Minimum anomaly size (pixels²)
--max-area 5000          # Maximum anomaly size (pixels²)
--max-annotations 3      # Max number of ML detections shown
--blue-threshold 30      # Max blue content (%) for ML detections
--thermal-threshold 200  # Temperature threshold (0-255)
```

---

## 📁 Project Structure

```
Arbit-Transformer-Analysis/
├── unified_thermal_analysis.py      # ⭐ Main script
├── run_unified_analysis.sh          # Quick launcher
├── batch_process.sh                 # Batch processor
├── UNIFIED_ANALYSIS_README.md       # Full docs
├── QUICKSTART.md                    # This file
├── TEST_RESULTS.md                  # Test summary
├── thermal_env/                     # Virtual environment
├── ML_analysis/
│   └── models/best_model.pth        # ML model
├── heat_point_analysis/
│   └── thermal_hotpoint_detector.py # Thermal detector
└── unified_results/                 # 📊 Output folder
    ├── *_combined_annotated.jpg     # ⭐ Main outputs
    ├── *_unified_analysis.png       # Visualizations
    └── *_unified_report.txt         # Reports
```

---

## 🔧 Common Tasks

### View Results
```bash
ls -lh unified_results/
```

### Read Report
```bash
cat unified_results/T1_faulty_001_unified_report.txt
```

### Open Output Image (macOS)
```bash
open unified_results/T1_faulty_001_combined_annotated.jpg
```

### Process Different Folder
```bash
./run_unified_analysis.sh Dataset/T2/faulty/T2_faulty_001.jpg
```

### Adjust Sensitivity
```bash
# More ML detections
python unified_thermal_analysis.py <image> --threshold 0.3 --max-annotations 5

# More thermal detections
python unified_thermal_analysis.py <image> --thermal-threshold 150
```

---

## 🐛 Troubleshooting

### Issue: `No module named 'torch'`
**Fix**: Activate virtual environment
```bash
source thermal_env/bin/activate
```

### Issue: No detections shown
**Fix**: Lower thresholds
```bash
python unified_thermal_analysis.py <image> --threshold 0.3 --thermal-threshold 150
```

### Issue: Too many detections
**Fix**: Increase thresholds
```bash
python unified_thermal_analysis.py <image> --threshold 0.7 --thermal-threshold 220
```

---

## 📊 Example Workflow

```bash
# 1. Activate environment
source thermal_env/bin/activate

# 2. Process one image
python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg

# 3. Check results
ls unified_results/

# 4. View report
cat unified_results/T1_faulty_001_unified_report.txt

# 5. Batch process more
./batch_process.sh Dataset/T1/faulty 10
```

---

## 💡 Pro Tips

1. **Use shell script for quick testing**: `./run_unified_analysis.sh <image>`
2. **Use batch script for multiple images**: `./batch_process.sh <folder> <count>`
3. **Lower `--threshold` for more ML detections** (default: 0.5, try: 0.3)
4. **Lower `--thermal-threshold` for more thermal hotspots** (default: 200, try: 150)
5. **Increase `--max-annotations` to show more results** (default: 3, try: 5)
6. **Main output is `*_combined_annotated.jpg`** - use this for reports!

---

## 📈 Parameters Cheat Sheet

| Want More... | Adjust This | Example |
|--------------|-------------|---------|
| ML detections | Lower `--threshold` | `--threshold 0.3` |
| Thermal hotspots | Lower `--thermal-threshold` | `--thermal-threshold 150` |
| Shown annotations | Increase `--max-annotations` | `--max-annotations 5` |
| Small detections | Lower `--min-area` | `--min-area 100` |
| Large detections | Increase `--max-area` | `--max-area 10000` |

---

## ✅ System Status

- ✅ ML Analysis: **WORKING**
- ✅ Thermal Analysis: **WORKING**
- ✅ Combined Output: **WORKING**
- ✅ Batch Processing: **WORKING**
- ✅ Documentation: **COMPLETE**

---

## 🎯 The Bottom Line

**One command, three outputs:**

```bash
./run_unified_analysis.sh Dataset/T1/faulty/T1_faulty_001.jpg
```

**You get:**
1. ⭐ `*_combined_annotated.jpg` - Your main result!
2. `*_unified_analysis.png` - Full visualization
3. `*_unified_report.txt` - Detailed report

**RED boxes** = ML anomalies  
**YELLOW boxes** = Thermal hotspots

**That's it!** 🎉

---

## 📚 More Information

- Full documentation: `UNIFIED_ANALYSIS_README.md`
- Test results: `TEST_RESULTS.md`
- Script source: `unified_thermal_analysis.py`

---

**Last Updated**: October 3, 2025  
**Status**: 🟢 Fully Operational
