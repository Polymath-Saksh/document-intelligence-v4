import re
from typing import Dict, Any
import spacy
from spacy.matcher import Matcher
import os

# Load spaCy English model (make sure to install: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Error loading spaCy model: {e}. Attempting to download...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Patterns for extracting age, procedure, location, and policy duration
AGE_PATTERN = re.compile(r"(\d{1,3})\s*[-]?(year|yr|y|yo|years|old|m|male|f|female)?", re.IGNORECASE)
DURATION_PATTERN = re.compile(r"(\d+)[-\s]*(month|year|day|week)[-\s]*(old|policy)?", re.IGNORECASE)

# List of common procedures (expand as needed)
COMMON_PROCEDURES = [
    "knee surgery", "hip replacement", "appendectomy", "bypass surgery", "angioplasty",
    "cataract surgery", "hernia repair", "gallbladder removal", "hysterectomy", "prostate surgery"
]

# Helper to extract procedure from text
def extract_procedure(text: str) -> str:
    text_lower = text.lower()
    for proc in COMMON_PROCEDURES:
        if proc in text_lower:
            return proc
    # Fallback: look for any noun chunk that could be a procedure
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        if "surgery" in chunk.text.lower() or "procedure" in chunk.text.lower():
            return chunk.text
    return ""

# Helper to extract location (GPE)
def extract_location(text: str) -> str:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return ""

# Helper to extract age
def extract_age(text: str) -> str:
    match = AGE_PATTERN.search(text)
    if match:
        return match.group(1)
    # Fallback: look for CARDINAL entity
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "CARDINAL" and int(ent.text) < 120:
            return ent.text
    return ""

# Helper to extract policy duration
def extract_policy_duration(text: str) -> str:
    match = DURATION_PATTERN.search(text)
    if match:
        return match.group(0)
    # Fallback: look for DATE entity
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DATE":
            return ent.text
    return ""

def parse_query(query: str) -> Dict[str, Any]:
    """
    Extracts structured fields from a natural language query.
    Returns a dict with keys: age, procedure, location, policy_duration
    """
    return {
        "age": extract_age(query),
        "procedure": extract_procedure(query),
        "location": extract_location(query),
        "policy_duration": extract_policy_duration(query)
    }
