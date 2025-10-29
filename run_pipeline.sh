#!/bin/bash

# ETL Pipeline Runner
# This script executes all three ETL scripts and saves outputs to separate files
# including both stdout and stderr

set -e  # Exit on error

echo "=========================================="
echo "Starting ETL Pipeline Execution"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Define output files
APPA_METEO_OUTPUT="$OUTPUT_DIR/ETL_appa-meteoTrentino_pipeline_output.txt"
EEA_OUTPUT="$OUTPUT_DIR/ETL_eea_pipeline_output.txt"
MERGE_OUTPUT="$OUTPUT_DIR/merge_datasets_output.txt"

echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================
# 1. Run APPA-meteoTrentino Pipeline
# ============================================
echo "----------------------------------------"
echo "Running: ETL_appa-meteoTrentino_pipeline.py"
echo "Output: $APPA_METEO_OUTPUT"
echo "----------------------------------------"
python3 "$SCRIPT_DIR/ETL_appa-meteoTrentino_pipeline.py" 2>&1 | tee "$APPA_METEO_OUTPUT"
echo ""

# ============================================
# 2. Run EEA Pipeline
# ============================================
echo "----------------------------------------"
echo "Running: ETL_eea_pipeline.py"
echo "Output: $EEA_OUTPUT"
echo "----------------------------------------"
python3 "$SCRIPT_DIR/ETL_eea_pipeline.py" 2>&1 | tee "$EEA_OUTPUT"
echo ""

# ============================================
# 3. Run Merge Datasets
# ============================================
echo "----------------------------------------"
echo "Running: merge_datasets.py"
echo "Output: $MERGE_OUTPUT"
echo "----------------------------------------"
python3 "$SCRIPT_DIR/merge_datasets.py" 2>&1 | tee "$MERGE_OUTPUT"
echo ""

# ============================================
# Pipeline Completion Summary
# ============================================
echo "=========================================="
echo "ETL Pipeline Execution Completed"
echo "=========================================="
echo ""
echo "Output files created:"
echo "  - $APPA_METEO_OUTPUT"
echo "  - $EEA_OUTPUT"
echo "  - $MERGE_OUTPUT"
echo ""
echo "To view outputs:"
echo "  cat $APPA_METEO_OUTPUT"
echo "  cat $EEA_OUTPUT"
echo "  cat $MERGE_OUTPUT"
echo ""
