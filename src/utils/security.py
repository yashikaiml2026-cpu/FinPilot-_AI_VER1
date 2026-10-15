from __future__ import annotations

import re
from typing import Any, Iterable


def mask_account_number(value: str | int | None) -> str:
    text = str(value or "")
    digits = re.sub(r"\D", "", text)
    if len(digits) <= 4:
        return "*" * max(len(digits), 1) + digits[-4:]
    return "********" + digits[-4:]


def mask_sensitive_value(value: str | int | None, visible_chars: int = 4) -> str:
    text = str(value or "")
    if len(text) <= visible_chars:
        return "*" * len(text)
    return text[:2] + "*" * (len(text) - visible_chars - 2) + text[-visible_chars:]


def sanitize_input(value: str | Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.strip()[:500]


def validate_required_fields(payload: dict[str, Any], required_fields: Iterable[str]) -> list[str]:
    missing = []
    for field in required_fields:
        if payload.get(field) in (None, "", []) or (isinstance(payload.get(field), str) and not payload.get(field).strip()):
            missing.append(field)
    return missing
