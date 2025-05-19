import fitz
import re
import spacy
from src.utils.text_cleaner import clean_text

nlp = spacy.load("en_core_web_sm")


async def parse_resume_text(file):
    # Read PDF
    contents = await file.read()
    # write temp and extract
    with open("temp.pdf", "wb") as f:
        f.write(contents)
    doc = fitz.open("temp.pdf")
    text = "".join([page.get_text() for page in doc])
    text = clean_text(text)
    return text
