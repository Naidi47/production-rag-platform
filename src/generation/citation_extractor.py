import re
from uuid import UUID

CITATION_PATTERN = re.compile(r"\[Source:\s*([0-9a-fA-F-]{36})\]")


def extract_citations(text: str) -> tuple[str, list[UUID]]:
    ids = []
    for match in CITATION_PATTERN.finditer(text):
        try:
            ids.append(UUID(match.group(1)))
        except ValueError:
            continue
    clean = CITATION_PATTERN.sub("", text)
    clean = re.sub(r"[ \t]+\n", "\n", clean).strip()
    return clean, list(dict.fromkeys(ids))
