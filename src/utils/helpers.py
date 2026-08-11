import re
import uuid


def generate_uuid() -> uuid.UUID:
    return uuid.uuid4()


def sanitize_text(text: str) -> str:
    # Remove control characters except newline, carriage-return and tab
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)