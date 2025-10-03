#!/bin/bash

echo "=========================================="
echo "API Comprehensive Test"
echo "=========================================="

mkdir -p test_results

# Test 1: T1 faulty (should have both ML + Thermal)
echo -e "\n[1/4] Testing T1 faulty image..."
curl -s -X POST \
  -F "baseline=@Dataset/T1/normal/T1_normal_001.jpg" \
  -F "maintenance=@Dataset/T1/faulty/T1_faulty_001.jpg" \
  -F "transformer_id=T1_Test" \
  -F "return_format=json" \
  http://localhost:8000/detect | python -m json.tool | grep -E "(total_anomalies|type)" | head -10

# Test 2: T2 faulty (should have thermal)
echo -e "\n[2/4] Testing T2 faulty image..."
curl -s -X POST \
  -F "baseline=@Dataset/T2/normal/T2_normal_001.png" \
  -F "maintenance=@Dataset/T2/faulty/T2_faulty_001.png" \
  -F "transformer_id=T2_Test" \
  -F "return_format=json" \
  http://localhost:8000/detect | python -m json.tool | grep -E "(total_anomalies|type)" | head -10

# Test 3: T3 faulty (should have ML anomalies)
echo -e "\n[3/4] Testing T3 faulty image..."
curl -s -X POST \
  -F "baseline=@Dataset/T3/normal/T3_normal_001.png" \
  -F "maintenance=@Dataset/T3/faulty/T3_faulty_001.png" \
  -F "transformer_id=T3_Test" \
  -F "return_format=json" \
  http://localhost:8000/detect | python -m json.tool | grep -E "(total_anomalies|type)" | head -10

# Test 4: Generate annotated images
echo -e "\n[4/4] Generating annotated images..."

curl -s -X POST \
  -F "baseline=@Dataset/T1/normal/T1_normal_001.jpg" \
  -F "maintenance=@Dataset/T1/faulty/T1_faulty_005.jpg" \
  -F "transformer_id=T1_005" \
  -F "return_format=annotated" \
  http://localhost:8000/detect \
  -o test_results/T1_005_annotated.png

echo "  ✓ Saved T1_005_annotated.png"

curl -s -X POST \
  -F "baseline=@Dataset/T2/normal/T2_normal_001.png" \
  -F "maintenance=@Dataset/T2/faulty/T2_faulty_005.png" \
  -F "transformer_id=T2_005" \
  -F "return_format=annotated" \
  http://localhost:8000/detect \
  -o test_results/T2_005_annotated.png

echo "  ✓ Saved T2_005_annotated.png"

echo -e "\n=========================================="
echo "All tests completed!"
echo "=========================================="
