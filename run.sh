#!/bin/bash
# CapCheck - One-Click Launcher

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "              CAPCHECK LAUNCHER                             "
echo "   Real vs. AI-Generated Media Forensics & Explainability   "
echo "============================================================"

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "[*] Creating Python virtual environment..."
    python3 -m venv venv
    echo "[*] Installing dependencies..."
    ./venv/bin/pip install -r requirements.txt
fi

# Ensure test samples exist
if [ ! -f "samples/real_sample.jpg" ]; then
    echo "[*] Generating bundled test media samples..."
    ./venv/bin/python3 generate_samples.py
fi

echo ""
echo "[*] Launching CapCheck Web Dashboard..."
echo "    URL: http://127.0.0.1:5001"
echo "    Press Ctrl+C to stop the server."
echo "============================================================"
echo ""

./venv/bin/python3 app.py
