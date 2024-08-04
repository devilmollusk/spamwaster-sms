#!/bin/bash

# Paths to the Python script and log file
VENV_PATH="/path/to/spamwaster-env"
PYTHON_SCRIPT_PATH="spamwaster-telegram.py"
LOG_FILE_PATH="output.log"

# Check if the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    # Activate the virtual environment
    source $VENV_PATH/bin/activate
fi
# Command to run the Python script in the background and redirect output to the log file
nohup python3 $PYTHON_SCRIPT_PATH > $LOG_FILE_PATH 2>&1 &

echo "Python script is running and logging to $LOG_FILE_PATH"
