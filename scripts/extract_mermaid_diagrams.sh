#!/bin/bash
# Extract Mermaid diagrams from progress report markdown

echo "=========================================="
echo "Mermaid Diagram Extractor"
echo "=========================================="
echo ""

REPORT_FILE="docs/APS360_Progress_Report_CONDENSED.md"
OUTPUT_DIR="figures/mermaid_source"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📄 Reading: $REPORT_FILE"
echo "📁 Output: $OUTPUT_DIR/"
echo ""

# Extract all mermaid diagrams using awk
echo "Extracting diagrams from condensed report..."
awk '/```mermaid/,/```/ {if (!/```/) print}' "$REPORT_FILE" > "$OUTPUT_DIR/all_diagrams_temp.txt"

# Split into individual diagrams
awk '/^graph LR/ || /^graph TB/ {c++} c==1' "$OUTPUT_DIR/all_diagrams_temp.txt" > "$OUTPUT_DIR/01_system_overview.mmd"
awk '/^graph LR/ || /^graph TB/ {c++} c==2' "$OUTPUT_DIR/all_diagrams_temp.txt" > "$OUTPUT_DIR/02_baseline_model.mmd"
awk '/^graph TB/ {c++} c==1' "$OUTPUT_DIR/all_diagrams_temp.txt" > "$OUTPUT_DIR/03_multimodal_architecture.mmd"

rm "$OUTPUT_DIR/all_diagrams_temp.txt"

echo "   ✅ Saved: $OUTPUT_DIR/01_system_overview.mmd (Section 1)"
echo "   ✅ Saved: $OUTPUT_DIR/02_baseline_model.mmd (Section 3)"
echo "   ✅ Saved: $OUTPUT_DIR/03_multimodal_architecture.mmd (Section 4)"

echo ""
echo "=========================================="
echo "📊 Next Steps:"
echo "=========================================="
echo ""
echo "Option 1: Convert using Mermaid Live Editor (Easiest)"
echo "  1. Go to: https://mermaid.live/"
echo "  2. Copy contents of each .mmd file"
echo "  3. Click 'Download SVG' or 'Download PNG'"
echo "  4. Save to figures/ directory"
echo ""
echo "Option 2: Use mermaid-cli (Automated)"
echo "  1. Install: npm install -g @mermaid-js/mermaid-cli"
echo "  2. Run conversion:"
echo "     mmdc -i $OUTPUT_DIR/01_system_overview.mmd -o figures/01_system_overview.png -w 1200 -H 600"
echo "     mmdc -i $OUTPUT_DIR/02_baseline_model.mmd -o figures/02_baseline_model.png -w 1200 -H 400"
echo "     mmdc -i $OUTPUT_DIR/03_multimodal_architecture.mmd -o figures/03_multimodal_architecture.png -w 1000 -H 1400"
echo ""
echo "✅ Mermaid source files extracted successfully!"
echo "=========================================="
