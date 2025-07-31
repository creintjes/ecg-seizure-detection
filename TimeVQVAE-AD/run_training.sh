#!/bin/bash

echo "Starting Stage 1 training..."
python stage1.py --dataset_ind 1,2

# Check if Stage 1 completed successfully
if [ $? -eq 0 ]; then
    echo "Stage 1 completed. Starting Stage 2..."
    python stage2.py --dataset_ind 1,2
    
    # Check if Stage 2 completed successfully
    if [ $? -eq 0 ]; then
        echo "Stage 2 completed. Starting evaluation..."
        python evaluation.py --dataset_ind 1,2
        if [ $? -eq 0 ]; then
            echo "All stages completed successfully!"
        else
            echo "Evaluation failed."
            exit 1
        fi
    else
        echo "Stage 2 failed. Stopping sequence."
        exit 1
    fi
else
    echo "Stage 1 failed. Stopping sequence."
    exit 1
fi