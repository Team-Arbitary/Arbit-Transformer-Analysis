#!/bin/bash
# API Testing Script
# Tests all endpoints of the Thermal Anomaly Detection API

API_URL="http://localhost:8000"
RESULTS_DIR="api_test_results"

echo "========================================"
echo "Thermal Anomaly Detection API - Test Suite"
echo "========================================"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Test 1: Health Check
echo "[1/5] Testing health endpoint..."
response=$(curl -s "$API_URL/health")
status=$(echo "$response" | python -m json.tool 2>/dev/null | grep -q '"status": "healthy"' && echo "✓ PASS" || echo "✗ FAIL")
echo "  $status - Health check"
echo ""

# Test 2: Config Endpoint
echo "[2/5] Testing config endpoint..."
response=$(curl -s "$API_URL/config")
status=$(echo "$response" | python -m json.tool 2>/dev/null | grep -q '"status": "success"' && echo "✓ PASS" || echo "✗ FAIL")
echo "  $status - Config retrieval"
echo ""

# Test 3: JSON Detection
echo "[3/5] Testing detect endpoint (JSON format)..."
response=$(curl -s -X POST \
  -F "baseline=@Dataset/T1/normal/T1_normal_001.jpg" \
  -F "maintenance=@Dataset/T1/faulty/T1_faulty_001.jpg" \
  -F "transformer_id=Test_JSON" \
  -F "return_format=json" \
  "$API_URL/detect")

status=$(echo "$response" | python -m json.tool 2>/dev/null | grep -q '"status": "success"' && echo "✓ PASS" || echo "✗ FAIL")
echo "  $status - JSON response format"

# Save JSON response
echo "$response" | python -m json.tool > "$RESULTS_DIR/test_json_response.json" 2>/dev/null
echo "  → Saved to: $RESULTS_DIR/test_json_response.json"
echo ""

# Test 4: Annotated Image Detection
echo "[4/5] Testing detect endpoint (Annotated image format)..."
curl -s -X POST \
  -F "baseline=@Dataset/T1/normal/T1_normal_001.jpg" \
  -F "maintenance=@Dataset/T1/faulty/T1_faulty_001.jpg" \
  -F "transformer_id=Test_Annotated" \
  -F "return_format=annotated" \
  "$API_URL/detect" \
  -o "$RESULTS_DIR/test_annotated.png" 2>/dev/null

if [ -f "$RESULTS_DIR/test_annotated.png" ] && [ -s "$RESULTS_DIR/test_annotated.png" ]; then
    echo "  ✓ PASS - Annotated image generated"
    echo "  → Saved to: $RESULTS_DIR/test_annotated.png"
else
    echo "  ✗ FAIL - Annotated image not generated"
fi
echo ""

# Test 5: Complete Format
echo "[5/5] Testing detect endpoint (Complete format)..."
response=$(curl -s -X POST \
  -F "baseline=@Dataset/T1/normal/T1_normal_001.jpg" \
  -F "maintenance=@Dataset/T1/faulty/T1_faulty_001.jpg" \
  -F "transformer_id=Test_Complete" \
  -F "return_format=complete" \
  "$API_URL/detect")

status=$(echo "$response" | python -m json.tool 2>/dev/null | grep -q '"annotated_image_base64"' && echo "✓ PASS" || echo "✗ FAIL")
echo "  $status - Complete format (JSON + base64 image)"

# Save complete response
echo "$response" | python -m json.tool > "$RESULTS_DIR/test_complete_response.json" 2>/dev/null
echo "  → Saved to: $RESULTS_DIR/test_complete_response.json"
echo ""

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "All test results saved to: $RESULTS_DIR/"
echo ""
echo "Generated files:"
ls -lh "$RESULTS_DIR/" | tail -n +2
echo ""
echo "========================================"
echo "API Testing Complete!"
echo "========================================"
