import re

# Header names (lowercase) that must be redacted in logs and error responses
_SENSITIVE_HEADER_PATTERNS = [
    "authorization",
    "x-api-key",
    "x-token",
    "cookie",
    "set-cookie",
    "proxy-authorization",
]

# Compiled regex for matching sensitive header names (case-insensitive)
_SENSITIVE_RE = re.compile(
    "|".join(re.escape(p) for p in _SENSITIVE_HEADER_PATTERNS),
    re.IGNORECASE,
)


_REDACT_PREFIX_LEN = 4


def _redact_value(value: str) -> str:
    """Keep first few chars of a secret for identification, mask the rest."""
    if len(value) <= _REDACT_PREFIX_LEN:
        return "[REDACTED]"
    return value[:_REDACT_PREFIX_LEN] + "...[REDACTED]"


def redact_headers(headers: dict) -> dict:
    """Return a copy of *headers* with sensitive values partially masked.

    Shows the first 4 characters of the value for identification, then
    replaces the rest with '[REDACTED]'.  Matching is case-insensitive
    against a known list of auth-related header names.
    The original dict is **never** mutated.
    """
    redacted = {}
    for key, value in headers.items():
        if _SENSITIVE_RE.fullmatch(key):
            redacted[key] = _redact_value(str(value))
        else:
            redacted[key] = value
    return redacted
