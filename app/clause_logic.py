import re
from typing import List


def match_clauses(text_chunks: List[str], questions: List[str], parsed_queries: List[dict]) -> List[str]:
    """
    Enhanced clause matching: uses structured fields (age, procedure, location, policy_duration) for more accurate retrieval.
    Returns the best-matching chunk for each question.
    """
    answers = []
    for i, question in enumerate(questions):
        pq = parsed_queries[i] if i < len(parsed_queries) else {}
        best_score = 0
        best_chunk = None
        for chunk in text_chunks:
            score = 0
            # Score by presence of structured fields
            if pq.get("age") and pq["age"] in chunk:
                score += 1
            if pq.get("procedure") and pq["procedure"].lower() in chunk.lower():
                score += 2
            if pq.get("location") and pq["location"].lower() in chunk.lower():
                score += 1
            if pq.get("policy_duration") and pq["policy_duration"].lower() in chunk.lower():
                score += 1
            # Also score by keyword overlap with question
            for word in question.split():
                if word.lower() in chunk.lower():
                    score += 0.2
            if score > best_score:
                best_score = score
                best_chunk = chunk
        if best_chunk:
            answers.append(best_chunk)
        else:
            answers.append("Clause not found in document.")
    return answers

# Add more advanced logic evaluation as needed for domain-specific scenarios
