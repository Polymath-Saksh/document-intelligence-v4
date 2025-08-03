import re
from typing import Dict, List, Any

# Define a set of regex patterns to identify contact details
# These are robust and can handle a variety of formats.
EMAIL_PATTERN = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
PHONE_PATTERN = r'(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?|\d{2,4}[\s-]?)?\d{3,4}[\s-]?\d{3,4}'
ADDRESS_PATTERN = r'(?:address|addr)[^\n\r:]*[:\-]?\s*(.*)' # This is a simple pattern, can be improved.

# A list of keyword patterns to detect questions about contact info.
CONTACT_QUESTION_KEYWORDS = [
    r'contact (details|information|info|email|phone|number|address|support)',
    r'how can i (contact|reach|get in touch|talk to|email|call|phone|find|connect with) ?(support|someone|person)?',
    r'who do i (contact|reach out to|email|call|phone|find|connect with) ?(support|someone|person)?',
    r'email address',
    r'phone number',
    r'contact person',
    r'support (email|number|contact|team|person|details|info|information)',
    r'reach (support|someone|person|team)',
    r'get in touch (with)? (support|someone|person|team)?',
    r'how do i (contact|reach|get in touch with|email|call|phone|find|connect with) ?(support|someone|person)?'
]

def is_contact_question(question: str) -> bool:
    """
    Checks if a question is likely asking for contact details using regex patterns.
    """
    q = question.lower()
    for pattern in CONTACT_QUESTION_KEYWORDS:
        if re.search(pattern, q, re.IGNORECASE):
            return True
    return False

def extract_contact_details(text: str) -> Dict[str, Any]:
    """
    Extracts all email addresses, phone numbers, and addresses from a given text.
    The address extraction is a simple heuristic and may need refinement for complex documents.
    """
    emails = list(set(re.findall(EMAIL_PATTERN, text)))
    phones = list(set(re.findall(PHONE_PATTERN, text)))
    
    # A more robust address extraction logic might be needed for different document types.
    # The current approach looks for 'address' and captures the rest of the line.
    addresses = list(set(re.findall(ADDRESS_PATTERN, text, re.IGNORECASE)))
    
    # Clean up addresses to remove empty or junk entries
    addresses = [addr.strip() for addr in addresses if addr.strip() and len(addr.strip()) > 5]

    return {
        'emails': emails,
        'phones': phones,
        'addresses': addresses
    }
