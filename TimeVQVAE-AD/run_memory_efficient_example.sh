#!/bin/bash

# Example script showing how to run memory-efficient evaluation

echo "🧠 Running Memory-Efficient Evaluation"
echo "======================================"

# Test with a small number of subjects first
echo "Testing with subjects 001-003..."

python evaluate_memory_efficient.py \
    --dataset_ind 001 002 003 \
    --memory_efficient \
    --max_memory_gb 6.0 \
    --chunk_size_minutes 20 \
    --overlap_minutes 3 \
    --reduced_parallel_processes 1 \
    --device 0

echo ""
echo "If this works well, you can scale up:"
echo ""
echo "# For more subjects with higher parallelization:"
echo "python evaluate_memory_efficient.py \\"
echo "    --dataset_ind 001 002 003 004 005 006 007 008 009 010 \\"
echo "    --memory_efficient \\"
echo "    --max_memory_gb 8.0 \\"
echo "    --chunk_size_minutes 30 \\"
echo "    --overlap_minutes 5 \\"
echo "    --reduced_parallel_processes 3 \\"
echo "    --device 0"
echo ""
echo "# Process ALL subjects (memory-efficient):"
echo "python evaluate_memory_efficient.py \\"
echo "    --dataset_ind all \\"
echo "    --memory_efficient \\"
echo "    --max_memory_gb 12.0 \\"
echo "    --chunk_size_minutes 45 \\"
echo "    --overlap_minutes 5 \\"
echo "    --reduced_parallel_processes 4 \\"
echo "    --device 0"
echo ""
echo "# Or run original method for comparison (single subject):"
echo "python evaluate_memory_efficient.py \\"
echo "    --dataset_ind 001 \\"
echo "    --device 0"
echo ""
echo "📋 NOTES:"
echo "- Overlap minutes prevents artifacts at chunk boundaries"
echo "- Results in overlap regions are properly averaged"
echo "- Monitor the overlap analysis output for transparency"
