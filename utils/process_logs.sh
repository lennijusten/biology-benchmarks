#!/bin/bash

# Check if a directory is provided
if [ $# -eq 0 ]; then
    echo "Please provide a directory path"
    exit 1
fi

LOG_DIR="$1"
SCRIPT_PATH="utils/calculate_cost.py"  # Update this path
MODEL_DATA_PATH="models/models_data.tsv"  # Update this path

# Check if the directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Directory does not exist: $LOG_DIR"
    exit 1
fi

# Loop through all .log files in the directory
for log_file in "$LOG_DIR"/*.json; do
    if [ -f "$log_file" ]; then
        echo "Processing file: $log_file"
        python "$SCRIPT_PATH" "$log_file" --model_data "$MODEL_DATA_PATH"
        echo "----------------------------------------"
    fi
done

echo "All log files processed."