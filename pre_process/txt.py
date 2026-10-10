"""Plain-text humor files -> documents.

A .txt file is either one continuous piece (a special's transcript, a book
chapter) or a collection of unrelated jokes, one per paragraph. The
difference matters for training: unrelated jokes must become separate
documents so they get an <|endoftext|> between them (Raschka §2.4).
"""
import re
from statistics import median

from .srt import read_text

CUE_PARAGRAPH = re.compile(r"^\s*[(\[][^)\]]{1,60}[)\]]\s*$")
AUDIENCE_WORDS = re.compile(r"laugh|applau|cheer|audience|crowd", re.I)
MAX_JOKE_CHARS = 400   # median paragraph length below this looks like a joke list
MIN_JOKES = 10


def paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def detect_layout(paras):
    """'jokes' for a collection of short, unrelated paragraphs, else 'document'.

    Transcripts give themselves away with stand-alone audience-reaction
    paragraphs like "(audience laughs)"; joke collections are many short
    paragraphs (and may still contain stage directions like "[sips water]").
    """
    if any(CUE_PARAGRAPH.match(p) and AUDIENCE_WORDS.search(p) for p in paras):
        return "document"
    if len(paras) >= MIN_JOKES and median(len(p) for p in paras) < MAX_JOKE_CHARS:
        return "jokes"
    return "document"


def to_documents(path, layout="auto"):
    """Return the documents in a .txt file.

    layout: "auto" (detect), "document" (whole file is one document) or
    "jokes" (every paragraph is its own document).
    """
    paras = paragraphs(read_text(path))
    if layout == "auto":
        layout = detect_layout(paras)
    if layout == "jokes":
        # A stand-alone stage direction sets up the joke that follows it.
        jokes, pending = [], ""
        for p in paras:
            if CUE_PARAGRAPH.match(p):
                pending = f"{pending} {p}".strip()
            else:
                jokes.append(f"{pending} {p}".strip())
                pending = ""
        return jokes
    return ["\n\n".join(paras)] if paras else []
