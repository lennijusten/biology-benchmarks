#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <prompt_schema>"
    echo "Example: $0 zero_shot_cot"
    exit 1
fi

PROMPT_SCHEMA=$1
BASE_DIR="$(pwd)/preprint/logs_for_sharing/$PROMPT_SCHEMA"
PYTHON_SCRIPT="$(pwd)/preprint/scripts/prepare-data/generate_result_csv_from_log_dir.py"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist"
    exit 1
fi

# Check if this is a CoT run based on prompt schema name
if [[ $PROMPT_SCHEMA == *"cot"* ]]; then
    COT_FLAG="--cot"
else
    COT_FLAG=""
fi

# Find all model directories and process them
find "$BASE_DIR" -mindepth 2 -maxdepth 2 -type d | while read -r model_dir; do
    # Check if directory contains any .json files
    if ls "$model_dir"/*.json 1> /dev/null 2>&1; then
        echo "Processing $model_dir..."
        python "$PYTHON_SCRIPT" --prompt_schema "$PROMPT_SCHEMA" $COT_FLAG "$model_dir"
    else
        echo "Skipping $model_dir - no .json files found"
    fi
done

echo "Processing complete!"