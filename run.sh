#!/usr/bin/env bash
# Quickstart script for Tehri Hydroelectric Operations Portal & Analytics

cd "$(dirname "$0")"

echo "======================================================================"
echo "  NATIONAL HYDROELECTRIC OPERATIONS & GRID DISPATCH PORTAL (NHOP)"
echo "  Government of India • Ministry of Power • CEA • THDC India Ltd"
echo "======================================================================"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not found in PATH."
    exit 1
fi

# Launch the unified portal
python3 run.py "$@"
