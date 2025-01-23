import re
from typing import Optional
def parse_cot_answer(completion: str) -> Optional[str]:
    """
    Extract answer from chain-of-thought completion using improved matching.
    Implements fix in issue #721 - https://github.com/UKGovernmentBEIS/inspect_ai/issues/721.
    First tries strict end-of-line matching, then falls back to looser matching.
    
    Args:
        completion: The model's response text
        
    Returns:
        The extracted answer if found (uppercase), otherwise None
    """
    # First try strict match - answer at end with only whitespace after
    match = re.search(
        r"(?i)^ANSWER\s*:\s*([A-Za-z ,]+)\s*(?:$|\n)",
        completion,
        flags=re.MULTILINE
    )
    
    # If no strict match, try less strict matching
    if match is None:
        match = re.search(
            r"(?i)ANSWER\s*:\s*([A-Za-z ,]+)(?:[^\w]|\n|$)", 
            completion
        )
    
    if match and match.group(1):
        return match.group(1).strip().upper()
    return None

def compute_cot_accuracy(samples: list) -> tuple[float, float]:
    """
    Compute accuracy and stderr for chain-of-thought responses.
    
    Args:
        samples: List of sample dictionaries from the log file
        
    Returns:
        Tuple of (accuracy, stderr)
    """
    correct_count = 0
    total_count = 0
    
    for sample in samples:
        # Get the model's response - handle list structure
        message_content = sample['messages'][1]['content']
        
        # Extract text from content structure
        if isinstance(message_content, list):
            for content_part in message_content:
                if isinstance(content_part, dict) and content_part.get('type') == 'text':
                    completion = content_part.get('text', '')
                    break
            else:
                print(f"Warning: No text content found in message: {message_content}")
                continue  # Skip if no text content found
        else:
            completion = message_content
            
        # Extract answer using improved parsing
        extracted_answer = parse_cot_answer(completion)
        
        if extracted_answer:
            # Compare extracted answer with target
            target = sample['target'].strip().upper()
            correct_count += (extracted_answer == target)
        total_count += 1
    
    # Compute accuracy and stderr
    accuracy = correct_count / total_count if total_count > 0 else 0
    stderr = (accuracy * (1 - accuracy) / total_count) ** 0.5 if total_count > 0 else 0
    
    return accuracy, stderr