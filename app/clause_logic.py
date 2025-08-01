import re
from typing import List

def match_clauses(text_chunks: List[str], questions: List[str]) -> List[str]:
    # Simple keyword-based clause matching for demonstration
    answers = []
    for question in questions:
        found = False
        for chunk in text_chunks:
            if any(word.lower() in chunk.lower() for word in question.split()):
                answers.append(chunk)
                found = True
                break
        if not found:
            answers.append("Clause not found in document.")
    return answers

# Add more advanced logic evaluation as needed for domain-specific scenarios
