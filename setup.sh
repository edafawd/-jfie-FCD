#!/usr/bin/env bash
# Setup for robot_brain.py on Raspberry Pi 5
set -e

sudo apt update
sudo apt install -y python3-opencv python3-pip

pip install ultralytics pyserial flask requests --break-system-packages

# --- Ollama (optional, for the AI narration) ---
# Best run on your PC (faster) and point OLLAMA_HOST in robot_brain.py at it.
# To run it on the Pi instead, uncomment:
# curl -fsSL https://ollama.com/install.sh | sh
# ollama pull moondream

echo "Setup done. Run:  python3 robot_brain.py"
