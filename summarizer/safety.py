"""Small, deterministic safeguards for values that may be retained or shown."""

from __future__ import annotations

import re

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{36,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"(?i)\bauthorization\s*:\s*basic\s+[^\s,;]+"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"://[^\s/@:]+:[^\s/@]+@"),
)


def redact_text(value: str) -> str:
    """Replace common credential forms without changing unrelated prose."""
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
