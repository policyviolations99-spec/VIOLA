"""
Text processing utilities for LLM call analysis.

This module provides utilities for:
1. Text cleaning and normalization
2. Stemming for error detection
3. JSON and code parsing
4. Format validation
"""

import re
import json
import logging
from typing import Any, Dict, Optional, Tuple
from nltk.stem import PorterStemmer

logger = logging.getLogger(__name__)

# Initialize stemmer (lazy loaded)
_stemmer = None


def get_stemmer() -> PorterStemmer:
    """Get or initialize the Porter stemmer."""
    global _stemmer
    if _stemmer is None:
        try:
            _stemmer = PorterStemmer()
        except Exception as e:
            logger.warning(f"Failed to initialize stemmer: {e}. Using simple lowercase fallback.")
            _stemmer = None
    return _stemmer


def stem_word(word: str) -> str:
    """
    Stem a single word.

    Args:
        word: Word to stem

    Returns:
        Stemmed word (lowercase)
    """
    stemmer = get_stemmer()
    if stemmer:
        try:
            return stemmer.stem(word.lower())
        except:
            return word.lower()
    return word.lower()


def clean_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Clean and normalize text.

    Args:
        text: Input text
        max_length: Maximum length to truncate to (None for no truncation)

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length]

    return text


def extract_json_from_markdown(text: str) -> Optional[str]:
    """
    Extract JSON content from markdown code blocks.

    Handles formats like:
```json
    {"key": "value"}
```

    Args:
        text: Text potentially containing markdown JSON

    Returns:
        Extracted JSON string or None
    """
    # Pattern for ```json ... ```
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        return matches[0].strip()

    return None


def parse_json(text: str, strict: bool = False) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Attempt to parse JSON from text.

    Tries multiple strategies:
    1. Direct JSON parsing
    2. Extract from markdown blocks
    3. Find JSON-like structures

    Args:
        text: Text to parse
        strict: If True, only try direct parsing

    Returns:
        Tuple of (parsed_dict, success)
    """
    if not text:
        return None, False

    # Strategy 1: Direct parsing
    try:
        parsed = json.loads(text)
        return parsed, True
    except json.JSONDecodeError:
        if strict:
            return None, False

    # Strategy 2: Extract from markdown
    json_content = extract_json_from_markdown(text)
    if json_content:
        try:
            parsed = json.loads(json_content)
            return parsed, True
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find JSON-like structure (starts with { or [)
    # Look for first { or [ and try to parse from there
    for start_char in ['{', '[']:
        start_idx = text.find(start_char)
        if start_idx != -1:
            try:
                parsed = json.loads(text[start_idx:])
                return parsed, True
            except json.JSONDecodeError:
                continue

    return None, False


def extract_code_from_markdown(text: str, language: Optional[str] = None) -> Optional[str]:
    """
    Extract code from markdown code blocks.

    Args:
        text: Text potentially containing markdown code
        language: Specific language to look for (e.g., 'python'), or None for any

    Returns:
        Extracted code string or None
    """
    if language:
        pattern = f'```{language}\\s*\\n?(.*?)\\n?```'
    else:
        pattern = r'```(?:\w+)?\s*\n?(.*?)\n?```'

    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Return the first match
        return matches[0].strip()

    return None


def is_valid_python_code(code: str) -> bool:
    """
    Check if code is syntactically valid Python.

    Args:
        code: Python code string

    Returns:
        True if valid, False otherwise
    """
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def contains_error_stems(text: str) -> bool:
    """
    Check if text contains error-related words (stemmed).

    Looks for stems of: error, fail, exception, invalid, etc.

    Args:
        text: Text to check

    Returns:
        True if error stems found
    """
    if not text:
        return False

    # Error keywords to check (will be stemmed)
    error_keywords = [
        'error', 'fail', 'failure', 'failed',
        'exception', 'invalid', 'incorrect',
        'unable', 'cannot', 'could not'
    ]

    # Stem the keywords
    error_stems = {stem_word(word) for word in error_keywords}

    # Tokenize text (simple word splitting)
    words = re.findall(r'\b\w+\b', text.lower())

    # Stem each word and check
    for word in words:
        if stem_word(word) in error_stems:
            return True

    return False


def is_truncated(text: str, max_length: int = 10000) -> bool:
    """
    Check if text appears to be truncated.

    Heuristics:
    1. Ends with ellipsis (...)
    2. Ends mid-sentence (no proper punctuation)
    3. Very long (likely hit token limit)
    4. Unclosed brackets/braces

    Args:
        text: Text to check
        max_length: Consider truncated if longer than this

    Returns:
        True if text appears truncated
    """
    if not text:
        return False

    text = text.strip()

    # Check 1: Ends with ellipsis
    if text.endswith('...') or text.endswith('…'):
        return True

    # Check 2: Very long text (likely hit limit)
    if len(text) > max_length:
        return True

    # Check 3: Unclosed brackets/braces in JSON-like content
    if text.startswith('{') or text.startswith('['):
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        if open_braces > 0 or open_brackets > 0:
            return True

    # Check 4: Ends mid-sentence (last 20 chars have no sentence-ending punctuation)
    tail = text[-20:] if len(text) > 20 else text
    if not any(punct in tail for punct in '.!?;"\')}]'):
        # But allow if it ends with common completions
        if not any(text.endswith(end) for end in ['```', '"""', "'''"]):
            return True

    return False


def extract_output_format(text: str) -> str:
    """
    Identify the format of output text.

    Returns:
        Format type: 'json', 'code', 'markdown', 'text'
    """
    if not text:
        return 'text'

    text = text.strip()

    # Check for JSON
    if text.startswith(('{', '[')):
        _, is_json = parse_json(text, strict=True)
        if is_json:
            return 'json'

    # Check for markdown code blocks
    if '```' in text:
        return 'markdown'

    # Check for code-like content (heuristic)
    code_indicators = ['def ', 'class ', 'import ', 'from ', 'return ', '    ']  # indentation
    if any(indicator in text for indicator in code_indicators):
        return 'code'

    return 'text'


def get_output_length(text: str) -> int:
    """
    Get the length of output text.

    For JSON, returns number of characters in stringified version.
    For code, returns number of lines.
    For text, returns number of characters.

    Args:
        text: Output text

    Returns:
        Length metric
    """
    if not text:
        return 0

    format_type = extract_output_format(text)

    if format_type == 'json':
        parsed, success = parse_json(text)
        if success:
            return len(json.dumps(parsed))
    elif format_type == 'code':
        return len(text.split('\n'))

    return len(text)