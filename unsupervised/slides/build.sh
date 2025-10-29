#!/bin/bash
# Build script for clustering lecture
# Regenerates all figures and compiles slides

set -e  # Exit on error

echo "======================================================================"
echo "BUILDING CLUSTERING LECTURE"
echo "======================================================================"

# Step 1: Generate all figures
echo ""
echo "[1/3] Generating figures from Python scripts..."
cd ../assets/clustering
python3 generate_figures.py

# Step 2: Copy figures to slides directory
echo ""
echo "[2/3] Copying figures to slides directory..."
cp -r figures ../../slides/

# Step 3: Compile Typst slides
echo ""
echo "[3/3] Compiling Typst slides..."
cd ../../slides
typst compile clustering-lecture.typ

echo ""
echo "======================================================================"
echo "BUILD COMPLETE!"
echo "======================================================================"
echo "Output: clustering-lecture.pdf"
ls -lh clustering-lecture.pdf
echo "======================================================================"
