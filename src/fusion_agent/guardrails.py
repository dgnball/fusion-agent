"""Checks applied around the model.

These are deliberately simple and testable. They don't make the model trustworthy. They make specific
failures visible and, where possible, recoverable:

- prompt injection in retrieved data is flagged before the model sees it
- every factual answer must cite sources, and every citation must be a source the agent actually retrieved
- answers that cite nothing must say plainly that the data doesn't cover the question
"""

import re

INJECTION_PATTERNS = [
    r"ignore (all |your |any )?(previous|prior|above) instructions",
    r"disregard (all |your |the )?(previous|prior|above)",
    r"(system|developer) (notice|message|prompt) to (ai|llm|assistant)",
    r"you are now",
    r"new instructions:",
]

CITATION = re.compile(r"\[([A-Z]{1,2}[-:][^\]]+)\]")
SOURCE_ID = re.compile(r"[A-Z]{1,2}[-:]\S+")
NO_DATA = "I don't have data to answer that"


def scan_for_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def citations(answer: str) -> list[str]:
    found = []
    for match in CITATION.findall(answer):
        for part in match.split(","):
            part = part.strip()
            if SOURCE_ID.fullmatch(part) and part not in found:
                found.append(part)
    return found


def check_answer(answer: str, retrieved: set[str]) -> list[str]:
    """Return a list of problems with the answer. An empty list means it passes."""
    cited = citations(answer)
    if not cited:
        if NO_DATA.lower() in answer.lower():
            return []
        return [
            "The answer cites no sources. Cite each claim like [R-005], or say "
            f'"{NO_DATA}." if the data doesn\'t cover the question.'
        ]
    unknown = [c for c in cited if c not in retrieved]
    if unknown:
        return [
            f"These citations were not returned by any tool in this run: {', '.join(unknown)}. "
            "Only cite source_id values you have seen in tool results."
        ]
    return []
