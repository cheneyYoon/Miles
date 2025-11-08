#!/bin/bash

# Local Development Setup Script
# Automates the setup process for running validation locally

set -e  # Exit on error

echo "======================================================================"
echo "Miles Project - Local Development Setup"
echo "======================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Must run from project root directory"
    echo "   Expected: /Users/cheneyyoon/Desktop/U of T/APS360/Miles/"
    echo "   Current:  $(pwd)"
    exit 1
fi

echo "✅ Running from project root"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "  $python_version"

# Extract version number
version_number=$(echo $python_version | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$version_number" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8 or higher required (found $version_number)"
    exit 1
fi

echo "  ✅ Python version OK"
echo ""

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists at: venv/"
    echo ""
    read -p "Do you want to remove it and create a fresh one? (y/n): " answer
    if [ "$answer" = "y" ]; then
        echo "Removing old virtual environment..."
        rm -rf venv
        echo "  ✅ Removed"
    else
        echo "Keeping existing virtual environment"
    fi
    echo ""
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "  ✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "  ✅ Activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "  ✅ Pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "  (This will take 3-5 minutes, please wait...)"
echo ""

pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "  ✅ All dependencies installed successfully!"
else
    echo "  ❌ Some dependencies failed to install"
    echo "  Try running manually: pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Verifying Installation"
echo "======================================================================"
echo ""

# Test imports
python3 << 'EOF'
import sys
errors = 0

packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers (BERT)',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'tqdm': 'tqdm'
}

for package, name in packages.items():
    try:
        __import__(package)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name} - FAILED")
        errors += 1

if errors > 0:
    print(f"\n❌ {errors} package(s) failed to import")
    sys.exit(1)
else:
    print("\n✅ All packages verified!")
    sys.exit(0)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation verification failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Setup Complete! 🎉"
echo "======================================================================"
echo ""
echo "What's next:"
echo ""
echo "1. Open your IDE (VS Code recommended):"
echo "   cd $(pwd)"
echo "   code ."
echo ""
echo "2. Open the validation notebook:"
echo "   notebooks/phase1_dry_run_local.ipynb"
echo ""
echo "3. Select the Python kernel:"
echo "   - In VS Code: Click kernel selector (top-right)"
echo "   - Choose: ./venv/bin/python"
echo ""
echo "4. Run all cells:"
echo "   - Click 'Run All' or press Shift+Enter on each cell"
echo ""
echo "5. Look for ✅ markers - all should pass!"
echo ""
echo "Alternative: Use Jupyter Lab:"
echo "   jupyter lab"
echo "   # Then open notebooks/phase1_dry_run_local.ipynb"
echo ""
echo "======================================================================"
echo ""
echo "Virtual environment location: $(pwd)/venv"
echo "To activate later: source venv/bin/activate"
echo ""
echo "Need help? See docs/LOCAL_SETUP.md for detailed instructions"
echo ""
