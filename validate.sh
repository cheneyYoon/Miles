#!/bin/bash

# Quick validation script for Miles project
# Run this before uploading to Colab

echo "=========================================="
echo "Miles Project - Quick Validation"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   Expected: /path/to/Miles/"
    echo "   Current:  $(pwd)"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "⚠️  No virtual environment found."
    echo "   Do you want to create one? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "✅ Virtual environment created"
        echo ""
        echo "Activating virtual environment..."
        source venv/bin/activate
        echo ""
        echo "Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
else
    echo "  ✅ Virtual environment found"
    echo ""
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "=========================================="
echo "Running validation tests..."
echo "=========================================="
echo ""

# Run the validation script
python3 tests/test_modules_quick.py

# Capture exit code
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "=========================================="
    echo "✅ VALIDATION SUCCESSFUL!"
    echo "=========================================="
    echo ""
    echo "Your code is ready for Colab! 🚀"
    echo ""
    echo "Next steps:"
    echo "1. Upload to Google Drive or GitHub"
    echo "2. Open notebooks/phase1_training_colab.ipynb"
    echo "3. Select A100 GPU runtime"
    echo "4. Run all cells"
    echo ""
else
    echo "=========================================="
    echo "❌ VALIDATION FAILED"
    echo "=========================================="
    echo ""
    echo "Please fix the errors above before running in Colab."
    echo "See tests/README.md for troubleshooting tips."
    echo ""
fi

exit $exit_code
