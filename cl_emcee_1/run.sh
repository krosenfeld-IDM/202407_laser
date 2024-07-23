#!/bin/bash
export HEADLESS=1
python3 -m idmlaser.measles
python3 analyze_waves.py
python3 analyze_lwps.py
python3 logprob.py
exit $?