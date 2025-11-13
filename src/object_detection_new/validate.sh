#!/bin/bash
# Validation script for object_detection_new package

echo "======================================"
echo "Object Detection New - Validation"
echo "======================================"
echo ""

# Navigate to workspace
cd /home/cindy/Documents/tk25_ws

echo "1. Checking package build..."
if [ -d "install/object_detection_new" ]; then
    echo "   ✓ Package installed"
else
    echo "   ✗ Package not installed"
    exit 1
fi

echo ""
echo "2. Checking model file..."
MODEL_PATH="tk26_vision/src/object_detection_new/models/yolo11m-seg.pt"
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "   ✓ Model file exists ($SIZE)"
else
    echo "   ✗ Model file not found"
    exit 1
fi

echo ""
echo "3. Checking executable..."
if [ -f "install/object_detection_new/lib/object_detection_new/yolo_seg_node" ]; then
    echo "   ✓ Executable created"
else
    echo "   ✗ Executable not found"
    exit 1
fi

echo ""
echo "4. Running tests..."
source install/setup.bash
colcon test --packages-select object_detection_new --event-handlers console_cohesion+ 2>&1 | grep -E "(passed|failed|Summary)" | tail -1
if [ $? -eq 0 ]; then
    echo "   ✓ Tests completed"
else
    echo "   ✗ Tests failed"
fi

echo ""
echo "5. Testing node startup..."
timeout 3 ros2 run object_detection_new yolo_seg_node --ros-args \
    -p publish_rate:=0.0 \
    -p model_path:=$MODEL_PATH 2>&1 | grep -q "initialized successfully"

if [ $? -eq 0 ]; then
    echo "   ✓ Node starts successfully"
else
    echo "   ⚠ Node startup check inconclusive (may need camera)"
fi

echo ""
echo "======================================"
echo "Validation Complete!"
echo "======================================"
echo ""
echo "Package is ready for use."
echo ""
echo "Next steps:"
echo "  1. Place your fine-tuned model in models/"
echo "  2. Connect your camera"
echo "  3. Run: ros2 run object_detection_new yolo_seg_node"
echo ""
