import re


def clean_text(text: str) -> str:
    # Basic cleanup
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_cgpa(text: str) -> float:
    match = re.search(r"CGPA[:\s]*([0-9]+\.?[0-9]*)", text, re.I)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None
