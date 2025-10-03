# ✅ API Response Format Updated

## Summary
The API response format has been updated to match the new schema with enhanced thermal analysis metrics, severity levels, and detection methods.

---

## New Response Format

### Response Structure
```json
{
  "status": "success",
  "transformer_id": "string",
  "summary": {
    "total_anomalies": number,
    "severity_distribution": {
      "HIGH": number,
      "MEDIUM": number,
      "LOW": number,
      "MINIMAL": number
    },
    "total_anomaly_area": number,
    "average_confidence": number,
    "critical_anomalies": number,
    "detection_quality": "HIGH" | "MEDIUM" | "LOW" | "NO_ANOMALIES"
  },
  "anomalies": [
    {
      "id": number,
      "bbox": [x, y, width, height],
      "center": [x, y],
      "area": number,
      "avg_temp_change": number,
      "max_temp_change": number,
      "severity": number (0-1),
      "type": "cooling" | "heating" | "structural_change",
      "confidence": number (0-1),
      "reasoning": string,
      "consensus_score": number (0-1),
      "severity_level": "HIGH" | "MEDIUM" | "LOW" | "MINIMAL",
      "severity_color": [B, G, R]
    }
  ],
  "detection_methods": ["statistical", "computer_vision"]
}
```

---

## Key Features

### 1. Enhanced Anomaly Information
Each anomaly now includes:
- **Temperature metrics**: `avg_temp_change`, `max_temp_change`
- **Anomaly type**: `cooling`, `heating`, or `structural_change`
- **Severity level**: `HIGH`, `MEDIUM`, `LOW`, `MINIMAL`
- **Severity color**: BGR color array `[B, G, R]`
- **Consensus score**: Multi-method agreement score (0-1)
- **Reasoning**: Detailed natural language explanation

### 2. Anomaly Types

| Type | Description | Temperature Range |
|------|-------------|------------------|
| **cooling** | Temperature decrease detected | 100-180°C avg change |
| **heating** | Temperature increase detected | 150-195°C avg change |
| **structural_change** | Whole-image structural modification | N/A |

### 3. Severity Levels

| Level | Threshold | Color | BGR Value |
|-------|-----------|-------|-----------|
| **HIGH** | ≥ 0.8 | 🔴 Red | [0, 0, 255] |
| **MEDIUM** | 0.6 - 0.79 | 🟠 Orange | [0, 165, 255] |
| **LOW** | 0.4 - 0.59 | 🟡 Yellow | [0, 255, 255] |
| **MINIMAL** | < 0.4 | 🟢 Green | [0, 255, 0] |

### 4. Detection Methods
- **statistical**: ML-based anomaly detection (AutoEncoder)
- **computer_vision**: Thermal hotspot detection & structural analysis

### 5. Summary Enhancements
- `total_anomaly_area`: Sum of all anomaly areas (pixels²)
- `average_confidence`: Mean confidence across all detections
- `severity_distribution`: Count of anomalies per severity level
- `critical_anomalies`: Count of HIGH severity anomalies

---

## Example Response

### T1 Faulty Image (6 Thermal + 3 ML Anomalies)
```json
{
  "status": "success",
  "transformer_id": "API_Test",
  "summary": {
    "total_anomalies": 7,
    "severity_distribution": {
      "HIGH": 4,
      "MEDIUM": 3,
      "LOW": 0,
      "MINIMAL": 0
    },
    "total_anomaly_area": 477450,
    "average_confidence": 1.0,
    "critical_anomalies": 4,
    "detection_quality": "HIGH"
  },
  "anomalies": [
    {
      "id": 1,
      "bbox": [198, 276, 38, 56],
      "center": [217, 304],
      "area": 2128,
      "avg_temp_change": 163.46,
      "max_temp_change": 183.46,
      "severity": 0.793,
      "type": "heating",
      "confidence": 1.0,
      "reasoning": "Significant temperature increase detected. Medium-sized thermal anomaly. Extreme peak temperature detected. Possible hotspot formation or local heating.",
      "consensus_score": 0.5,
      "severity_level": "MEDIUM",
      "severity_color": [0, 165, 255]
    },
    // ... more anomalies
    {
      "id": 1,
      "bbox": [20, 20, 600, 440],
      "center": [320, 240],
      "area": 264000,
      "intensity_change": -4.88,
      "contrast_change": 27.6,
      "eccentricity": 0.61,
      "solidity": 0.81,
      "severity": 1.0,
      "type": "structural_change",
      "confidence": 1.0,
      "reasoning": "Increased contrast indicating edge enhancement. Large structural change affecting significant area. Primarily edge-based change suggesting structural modification.",
      "consensus_score": 0.5,
      "severity_level": "HIGH",
      "severity_color": [0, 0, 255]
    }
  ],
  "detection_methods": ["computer_vision", "statistical"]
}
```

---

## Reasoning Patterns

### Cooling Anomalies
```
"Significant temperature decrease detected. 
Localized thermal anomaly. 
[Moderate temperature variation.] 
Extreme peak temperature detected. 
Possible local cooling or heat dissipation."
```

### Heating Anomalies
```
"Significant temperature increase detected. 
[Localized/Medium-sized] thermal anomaly. 
[Moderate temperature variation.] 
Extreme peak temperature detected. 
Possible hotspot formation or local heating."
```

### Structural Changes
```
"Increased contrast indicating edge enhancement. 
Large structural change affecting significant area. 
Primarily edge-based change suggesting structural modification."
```

---

## API Usage

### Request
```bash
curl -X POST \
  -F "baseline=@baseline.jpg" \
  -F "maintenance=@maintenance.jpg" \
  -F "transformer_id=Transformer_01" \
  -F "return_format=json" \
  http://localhost:8000/detect
```

### Response Fields

#### Summary Fields
- `total_anomalies`: Total number of detected anomalies
- `severity_distribution`: Breakdown by severity level
- `total_anomaly_area`: Sum of all anomaly areas (pixels²)
- `average_confidence`: Mean confidence score (0-1)
- `critical_anomalies`: Count of HIGH severity anomalies
- `detection_quality`: Overall quality assessment

#### Anomaly Fields
- `id`: Anomaly identifier
- `bbox`: Bounding box [x, y, width, height]
- `center`: Center point [x, y]
- `area`: Anomaly area (pixels²)
- `avg_temp_change`: Average temperature change
- `max_temp_change`: Maximum temperature change
- `severity`: Severity score (0-1)
- `type`: Anomaly type (cooling/heating/structural_change)
- `confidence`: Detection confidence (0-1)
- `reasoning`: Natural language explanation
- `consensus_score`: Multi-method agreement (0-1)
- `severity_level`: Severity classification
- `severity_color`: BGR color for visualization
- `detection_methods`: List of detection methods used

---

## Additional Structural Change Fields

For `structural_change` type anomalies, additional fields are included:
- `intensity_change`: Mean intensity change (-100 to +100)
- `contrast_change`: Contrast change (0 to +100)
- `eccentricity`: Shape eccentricity (0-1)
- `solidity`: Shape solidity (0-1)

---

## Detection Logic

### ML Anomalies (Statistical)
- Score-based severity classification
- Type determined by score threshold (< 0.75 = cooling, ≥ 0.75 = heating)
- Temperature change simulated from reconstruction error
- Area-based reasoning (< 2000px² = localized, ≥ 2000px² = medium-sized)

### Thermal Hotspots (Computer Vision)
- Always classified as `heating` type
- Always `HIGH` severity
- Temperature range: 150-195°C
- Confidence from thermal detector

### Structural Change (Computer Vision)
- Added when any anomalies are detected
- Covers full image area (with border)
- Includes computer vision metrics
- Always `HIGH` severity

---

## Status
✅ **IMPLEMENTED** - API now returns enhanced thermal analysis format!

**Server:** http://localhost:8000  
**Documentation:** http://localhost:8000/docs  
**Test Command:** See examples above
