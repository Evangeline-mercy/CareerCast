import spacy
import re

nlp = spacy.load("en_core_web_sm")

SKILL_LIST = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "html",
    "css",
    "javascript",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "pandas",
    "numpy",
    "tensorflow",
    "excel",
    "power bi",
    "tableau",
    "aws",
    "docker",
    "git",
    "linux",
    "vlsi",
    "embedded systems",
    "iot",
    "arduino",
    "verilog",
    "matlab"
]

EDUCATION_KEYWORDS = [
    "b.tech",
    "b.e",
    "bachelor",
    "m.tech",
    "mba",
    "mca",
    "bca",
    "b.sc",
    "m.sc",
    "phd",
    "diploma"
]


def parse_resume(text):

    if not isinstance(text, str):
        return {
            "skills": [],
            "education": [],
            "institutions": [],
            "roles": []
        }

    doc = nlp(text)
    lower = text.lower()

    skills_found = []

    for skill in SKILL_LIST:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, lower):
            skills_found.append(skill)

    education = [
        e for e in EDUCATION_KEYWORDS
        if e in lower
    ]

    institutions = [
        ent.text
        for ent in doc.ents
        if ent.label_ == "ORG"
    ]

    roles = [
        ent.text
        for ent in doc.ents
        if ent.label_ in ["ORG", "PERSON"]
    ]

    return {
        "skills": skills_found,
        "education": education,
        "institutions": institutions,
        "roles": roles
    }