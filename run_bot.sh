#!/bin/bash

# Paths to the Python script and log file
VENV_PATH="spamwaster"
PYTHON_SCRIPT_PATH="spamwaster-telegram.py"
PYTHON_FLASK_PATH="serve_logs"
LOG_FILE_PATH="output.log"
FLASK_LOG_FILE_PATH="flask.log"

# Check if the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    # Activate the virtual environment
    source $VENV_PATH/bin/activate
fi

# Command to run the Python script in the background and redirect output to the log file
nohup python3 -u $PYTHON_SCRIPT_PATH > $LOG_FILE_PATH 2>&1 &
# nohup exec gunicorn -w 1 -k eventlet -b 0.0.0.0:8000 $PYTHON_FLASK_PATH:app > $FLASK_LOG_FILE_PATH 2>&1 &

echo "Python script is running and logging to $LOG_FILE_PATH"
#echo "Flask app is running and logging to $FLASK_LOG_FILE_PATH"
