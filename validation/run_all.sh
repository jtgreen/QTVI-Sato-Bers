#!/bin/bash
################################################################################
# run_all.sh — Master orchestration script for the Sato-Bers validation suite.
#
# PURPOSE:
#   Automates the full validation pipeline:
#     1. Build the C++ reference exporter
#     2. Generate float64-precision reference traces (C++)
#     3. Run the Python validation suite against the reference
#     4. Run the Julia validation suite against the reference
#
# PREREQUISITES:
#   - g++ (C++23 capable)
#   - Python 3 with numpy
#   - Julia with StaticArrays package
#
# USAGE:
#   cd <project_root>
#   bash validation/run_all.sh
#
# EXIT CODES:
#   0 — All validations passed
#   1 — One or more validations failed
#   2 — Build or reference generation failed
################################################################################

set -e  # Exit immediately on error

# Navigate to project root (parent of validation/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "Sato-Bers Model Validation Suite"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# --------------------------------------------------------------------------
# Step 1: Build the C++ reference exporter
#
# We compile validate_export.cc together with cell.cc. The cell.cc file
# contains the ground-truth Sato-Bers model implementation.
# --------------------------------------------------------------------------
echo "--- Step 1: Building C++ reference exporter ---"
g++ -std=c++23 -O2 -I. -o validation/validate_export \
    validation/validate_export.cc cell.cc
echo "  Build successful."
echo ""

# --------------------------------------------------------------------------
# Step 2: Generate reference traces
#
# The C++ exporter runs all 6 scenarios and writes CSV files to
# validation/reference/ with full float64 precision (%.17g).
# --------------------------------------------------------------------------
echo "--- Step 2: Generating C++ reference traces ---"
./validation/validate_export
echo ""

# --------------------------------------------------------------------------
# Step 3: Run Python validation
#
# The Python script imports the Cell class from python/cell.py and
# reproduces all scenarios, comparing against the C++ reference CSVs.
# --------------------------------------------------------------------------
echo "--- Step 3: Running Python validation ---"
echo ""
PYTHON_EXIT=0
python3 validation/validate_python.py || PYTHON_EXIT=$?
echo ""

# --------------------------------------------------------------------------
# Step 4: Run Julia validation
#
# The Julia script loads SatoBers.jl and reproduces all scenarios,
# comparing against the same C++ reference CSVs.
# --------------------------------------------------------------------------
echo "--- Step 4: Running Julia validation ---"
echo ""
JULIA_EXIT=0
julia validation/validate_julia.jl || JULIA_EXIT=$?
echo ""

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------
echo "============================================================"
echo "FINAL RESULTS"
echo "============================================================"

EXIT_CODE=0

if [ $PYTHON_EXIT -eq 0 ]; then
    echo "  Python: ALL PASSED"
else
    echo "  Python: SOME FAILURES (exit code $PYTHON_EXIT)"
    EXIT_CODE=1
fi

if [ $JULIA_EXIT -eq 0 ]; then
    echo "  Julia:  ALL PASSED"
else
    echo "  Julia:  SOME FAILURES (exit code $JULIA_EXIT)"
    EXIT_CODE=1
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "All validations PASSED."
else
    echo "Some validations FAILED — see details above."
fi

exit $EXIT_CODE
