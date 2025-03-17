import json
import sys

def main(filename):
    total_statements = 0
    min_statements = None
    max_statements = None

    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue  # Skip empty lines
            try:
                question = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping invalid JSON line:", line)
                continue

            # Count all answer statements (regardless of correctness)
            count = len(question.get("answer_statements", []))
            total_statements += count

            if min_statements is None or count < min_statements:
                min_statements = count
            if max_statements is None or count > max_statements:
                max_statements = count

    if min_statements is not None:
        print("Total number of answer statements:", total_statements)
        print("Minimum answer statements in a question:", min_statements)
        print("Maximum answer statements in a question:", max_statements)
    else:
        print("No valid questions found in the file.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <jsonl_file>")
    else:
        main(sys.argv[1])

