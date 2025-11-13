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
APPA_METEO_OUTPUT="$OUTPUT_DIR/appa_meteo_merge_pipeline_output.txt"
EEA_FILTER_OUTPUT="$OUTPUT_DIR/eea_proximity_filter_pipeline_output.txt"
MERGE_PROXIMITY_OUTPUT="$OUTPUT_DIR/merge_appa_eea_proximity_output.txt"

echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================
# 1. Run APPA-Meteo Merge Pipeline
# ============================================
echo "----------------------------------------"
echo "Running: merge_appa_meteo_trentino.py"
echo "Output: $APPA_METEO_OUTPUT"
echo "----------------------------------------"
python3 "$SCRIPT_DIR/src/merge_appa_meteo_trentino.py" 2>&1 | tee "$APPA_METEO_OUTPUT"
echo ""

# ============================================
# 2. Run EEA Proximity Filter Pipeline
# ============================================
echo "----------------------------------------"
echo "Running: filter_eea_by_proximity.py"
echo "Output: $EEA_FILTER_OUTPUT"
echo "----------------------------------------"
python3 "$SCRIPT_DIR/src/filter_eea_by_proximity.py" 2>&1 | tee "$EEA_FILTER_OUTPUT"
echo ""

# ============================================
# 3. Run APPA-EEA Proximity Merge
# ============================================
echo "----------------------------------------"
echo "Running: merge_datasets_by_proximity.py"
echo "Output: $MERGE_PROXIMITY_OUTPUT"
echo "----------------------------------------"
python3 "$SCRIPT_DIR/src/merge_datasets_by_proximity.py" 2>&1 | tee "$MERGE_PROXIMITY_OUTPUT"
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
echo "  - $EEA_FILTER_OUTPUT"
echo "  - $MERGE_PROXIMITY_OUTPUT"
echo ""
echo "To view outputs:"
echo "  cat $APPA_METEO_OUTPUT"
echo "  cat $EEA_FILTER_OUTPUT"
echo "  cat $MERGE_PROXIMITY_OUTPUT"
echo ""