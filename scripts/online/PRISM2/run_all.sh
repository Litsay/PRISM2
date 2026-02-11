#!/bin/bash

# Run all PRISM2 experiments
# 运行所有PRISM2实验

echo "=========================================="
echo "Starting PRISM2 experiments"
echo "=========================================="

# PatchTST experiments
echo "Running PatchTST experiments..."
bash scripts/online/PRISM2/ETTh2.sh
bash scripts/online/PRISM2/ETTm1.sh
bash scripts/online/PRISM2/Weather.sh
bash scripts/online/PRISM2/ECL.sh
bash scripts/online/PRISM2/Traffic.sh

# iTransformer experiments
echo "Running iTransformer experiments..."
bash scripts/online/PRISM2/iTransformer_ETTh2.sh

# TCN experiments
echo "Running TCN experiments..."
bash scripts/online/PRISM2/TCN_ETTh2.sh

echo "=========================================="
echo "All PRISM2 experiments completed!"
echo "=========================================="
