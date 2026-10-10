"""SRT subtitles -> transcript text.

Only deals with the subtitle format: cue blocks, timestamps, markup, caption
line-wrapping and repeated captions. Humor-specific cleanup (sound cues,
speaker labels, normalization) happens in kasual_processor.
"""
import re
from dataclasses import dataclass
from pathlib import Path

TIMESTAMP = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})"
)
MARKUP = re.compile(r"</?[a-zA-Z][^>]*>|\{\\[^}]*\}")  # <i>, <font ...>, {\an8}


@dataclass
class Cue:
    start: float  # seconds
    end: float
    text: str


def read_text(path):
    """Read a subtitle/text file whatever its encoding (BOM, CRLF, cp1252)."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse(path):
    """Parse an .srt file into cues.

    Blocks are located by their timestamp line rather than the index line,
    because real files have broken indices (e.g. "e1").
    """
    cues = []
    for block in re.split(r"\n\s*\n", read_text(path)):
        lines = block.strip().split("\n")
        ts_idx = next((i for i, ln in enumerate(lines) if TIMESTAMP.search(ln)), None)
        if ts_idx is None:
            continue
        g = TIMESTAMP.search(lines[ts_idx]).groups()
        body = [MARKUP.sub("", ln).strip() for ln in lines[ts_idx + 1:]]
        text = " ".join(ln for ln in body if ln)
        if text:
            cues.append(Cue(_seconds(*g[:4]), _seconds(*g[4:]), text))
    return cues


def to_text(path, pause_s=2.0):
    """Join cues into a transcript.

    Caption fragments are merged into flowing lines, consecutive duplicate
    captions (rolling auto-captions) are dropped, and a silence longer than
    `pause_s` starts a new paragraph: in stand-up, those gaps are timing.
    """
    paragraphs, current, prev = [], [], None
    for cue in parse(path):
        if prev is not None and cue.text == prev.text:
            prev = cue
            continue
        if current and prev is not None and cue.start - prev.end > pause_s:
            paragraphs.append(" ".join(current))
            current = []
        current.append(cue.text)
        prev = cue
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def to_documents(path, pause_s=2.0):
    """A subtitle file is one performance, so one document."""
    text = to_text(path, pause_s)
    return [text] if text else []
