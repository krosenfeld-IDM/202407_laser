#!bin/bash
echo "Running idmlaser"
python3 -m idmlaser.measles
python3 analyze_waves.py
python3 analyze_lwps.py
echo "done"
