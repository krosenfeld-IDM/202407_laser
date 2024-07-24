#!bin/bash
echo "Running idmlaser"
python3 run_simulation.py
python3 analyze_waves.py
python3 analyze_lwps.py
echo "done"
exit $?