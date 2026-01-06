#!/bin/bash

# Start the Elliott Wave Analysis Application

set -e

echo "Starting Elliott Wave Analysis Application..."

# Check if running in virtual environment, activate if .venv exists
if [ -d ".venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the Flask application
python app_v5_automated.py
