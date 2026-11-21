"""
Text Preprocessing Module for SMS Spam Classification System
Provides text normalization, negation-preserving stop words, and robust input validation.
"""

import re
import html
from typing import Tuple, List, Set
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Critical negation words that must NOT be removed from SMS texts
NEGATION_WORDS = {
    "not", "no", "nor", "neither", "never", "none",
    "nobody", "nowhere", "nothing", "hardly", "scarcely",
    "barely", "cannot", "cant", "dont", "wont", "isnt",
    "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
    "doesnt", "didnt", "shouldnt", "couldnt", "wouldnt"
}

# Build custom stop word set preserving negation
CUSTOM_STOP_WORDS: List[str] = sorted(list(ENGLISH_STOP_WORDS - NEGATION_WORDS))


def clean_sms_text(text: str) -> str:
    """
    Cleans and normalizes raw SMS text.
    - Unescapes HTML entities (&lt;, &gt;, &amp;, etc.)
    - Converts text to lowercase
    - Normalizes multiple spaces, tabs, and newlines
    - Strips leading and trailing whitespaces
    """
    if not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    cleaned = html.unescape(text)
    
    # Convert to lowercase
    cleaned = cleaned.lower()
    
    # Replace tabs and newlines with spaces
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    
    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned


def validate_sms_input(text: str, max_length: int = 1000) -> Tuple[bool, str]:
    """
    Validates user input text for SMS classification.
    Rejects empty, whitespace-only, overly long, or non-alphanumeric inputs.
    
    Args:
        text: SMS message string
        max_length: Maximum allowed character length
        
    Returns:
        Tuple (is_valid, validation_message)
    """
    if text is None:
        return False, "Error: Message is empty. Please enter an SMS message."
    
    stripped = text.strip()
    if len(stripped) == 0:
        return False, "Error: Message contains only whitespace. Please enter valid text."
    
    if len(text) > max_length:
        return (
            False,
            f"Error: Message is too long ({len(text)} characters). Maximum limit is {max_length} characters."
        )
    
    # Check for presence of at least one alphanumeric character
    if not re.search(r"[a-zA-Z0-9]", stripped):
        return (
            False,
            "Error: Message must contain meaningful alphanumeric characters, not just punctuation or symbols."
        )
        
    return True, "Valid SMS input."


def get_custom_stop_words() -> List[str]:
    """Returns stop words list with negation words preserved."""
    return CUSTOM_STOP_WORDS
