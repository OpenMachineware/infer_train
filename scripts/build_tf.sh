#!/bin/bash
# scripts/build_tf.sh

set -e

echo "🧠 Building InferTrain TensorFlow plugin..."

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd pybinds/infer_train_tf

echo -e "${YELLOW}1. Building wheel with maturin...${NC}"
maturin build --release

WHEEL_PATH=$(ls ../../target/wheels/infer_train_tf-*.whl | head -1)
WHEEL_NAME=$(basename "$WHEEL_PATH")
FIXED_WHEEL_NAME="${WHEEL_NAME%.whl}_fixed.whl"
FIXED_WHEEL_PATH="$(cd ../.. && pwd)/target/wheels/$FIXED_WHEEL_NAME"

echo -e "${YELLOW}2. Extracting wheel...${NC}"
rm -rf tmp
unzip -q "$WHEEL_PATH" -d tmp

echo -e "${YELLOW}3. Replacing __init__.py...${NC}"
cp __init__.py tmp/infer_train_tf/

echo -e "${YELLOW}4. Repacking wheel...${NC}"
cd tmp
zip -q -r "../$FIXED_WHEEL_NAME" .
cd ..

echo -e "${YELLOW}5. Cleaning up...${NC}"
rm -rf tmp

echo -e "${GREEN}✅ Fixed wheel: $FIXED_WHEEL_PATH${NC}"
echo -e "${GREEN}   Install: pip install $FIXED_WHEEL_PATH --force-reinstall${NC}"