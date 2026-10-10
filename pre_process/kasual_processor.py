"""KasualProcessor: wraps the format preprocessors + tokenizer into a single interface.

Same shape as a Hugging Face multimodal processor (e.g. Kimi-K3's): each input
type has its own preprocessor (pre_process/srt.py, pre_process/txt.py), this
class applies the cleaning every format shares, and one __call__ hands the
resulting documents to the tokenizer and returns what the model consumes
(input_ids, attention_mask). It is an interface, not an entry point: finding
files, building corpora and reporting stats belong to the caller.
"""


import re
import unicodedata
from pathlib import Path
from transformers import AutoTokenizer
from . import srt, txt, KasualDataset

LAUGHTER, APPLAUSE, GROANS = "[laughter]", "[applause]", "[groans]"
MARKERS = {"laughter": LAUGHTER, "applause": APPLAUSE, "groans": GROANS}

CUE = re.compile(r"[(\[]([^()\[\]]{1,80})[)\]]")
LAUGHTER_WORDS = re.compile(r"laugh|chuckl|giggl|guffaw")
APPLAUSE_WORDS = re.compile(r"applau|cheer|clap|whistl|whoop|ovation")
GROAN_WORDS = re.compile(r"groan|moan|boo|gasp|disapprov|wince")
NOISE = re.compile(r"indistinct|chatter|conversation|music|tone|static|silence|^(man|woman|men|women)$")

CROWD = r"(?:man|woman|men|women|audience|crowd|all|announcer)(?:\s*\d+)?"
SPEAKER = re.compile(rf"(?:^|(?<=\s))-?\s*(?:({CROWD})|([A-Z][A-Z0-9 .'\-]{{1,30}})):\s+", re.I)
CROWD_LINE = re.compile(rf"(?:^|(?<=\s))-?\s*{CROWD}:\s+[^.!?]*[.!?]*", re.I)

TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
                            "–": "-", "—": " - ", " ": " ", "♪": "", "♫": ""})


def classify_cue(inner):
    """'laughter' | 'applause' | 'groans' | 'noise' | 'performer' for a cue's inner text.

    Order matters: impressions ("imitates club music") are performance even
    when they mention noise words, and anything the audience does is a
    reaction, not part of the act.
    """
    c = inner.lower().strip()
    audience = bool(re.match(r"(?:the\s+)?(?:audience|crowd)\b", c))
    if c.startswith(("imitat", "mimic", "impersonat")):
        return "performer"
    if LAUGHTER_WORDS.search(c) and (audience or re.match(r"(?:nervous\s+)?laughter", c)):
        return "laughter"
    if APPLAUSE_WORDS.search(c) and (audience or APPLAUSE_WORDS.match(c) or c.startswith("scattered")):
        return "applause"
    if audience and GROAN_WORDS.search(c):
        return "groans"
    if audience or NOISE.search(c):
        return "noise"
    return "performer"


class KasualProcessor:
    r"""Wraps the humor-text preprocessors and a tokenizer into a single processor.

    Args:
        tokenizer: a Hugging Face tokenizer (use `from_pretrained` to load one by name).
        audience: "marker" keeps [laughter]/[applause]/[groans] markers, "drop" removes them.
        performer: "keep" keeps performer cues as [action], "drop" removes them.
        pause_s: silence (seconds) that starts a new paragraph in subtitles.
        txt_layout: "auto" | "document" | "jokes" (see txt.detect_layout).
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer, audience="marker", performer="keep", pause_s=2.0, txt_layout="auto", manual_config=None):
        self.tokenizer : AutoTokenizer = tokenizer
        self.audience = audience
        self.performer = performer
        self.pause_s = pause_s
        self.txt_layout = txt_layout
        self.manual_config = manual_config

    @classmethod
    def from_pretrained(cls, name="gpt2",revision=None, **kwargs):
        return cls(AutoTokenizer.from_pretrained(name, trust_remote_code=True), **kwargs)

    # ── Format preprocessing ─────────────────────────────────────────────

    def preprocess(self, source):
        """One file -> its cleaned documents (a special is one; a joke list is many)."""
        path = Path(source)
        ext = path.suffix.lower()
        if ext == ".srt":
            documents = srt.to_documents(path, pause_s=self.pause_s)
        elif ext in (".txt", ".md"):
            documents = txt.to_documents(path, layout=self.txt_layout)
        else:
            raise ValueError(f"unsupported file type: {path.name}")
        return [d for d in map(self.clean, documents) if d]

    # ── Text cleaning (shared by every format) ───────────────────────────

    def _replace_cue(self, match):
        kind = classify_cue(match.group(1))
        if kind == "noise":
            return " "
        if kind in MARKERS:
            return f" {MARKERS[kind]} " if self.audience == "marker" else " "
        return f" [{match.group(1).strip().lower()}] " if self.performer == "keep" else " "

    def clean(self, text):
        text = unicodedata.normalize("NFKC", text.translate(TYPOGRAPHY))
        paragraphs = []
        for para in re.split(r"\n\s*\n", text):
            para = CROWD_LINE.sub(" ", para)
            para = SPEAKER.sub(" ", para)
            para = CUE.sub(self._replace_cue, para)
            para = re.sub(r"(^|\s)-\s*(?=[A-Za-z\"'\[.])", r"\1", para)  # dialogue dashes "-Yeah."
            para = re.sub(r"(^|\s)\"\s+(?=\w)", r'\1"', para)             # '" A, B' left by a removed ♪
            para = re.sub(r"(\[(?:laughter|applause|groans)\])(?:\s*\1)+", r"\1", para)  # repeats
            para = re.sub(r"\s+", " ", para).strip()
            para = re.sub(r"\s+([,!?;:]|\.(?!\.))", r"\1", para)  # "word ," -> "word," but keep " ...shows"
            if para and not re.fullmatch(r"[\W_]*", para):
                paragraphs.append(para)
        return "\n\n".join(paragraphs)

    # ── Main entry points ────────────────────────────────────────────────

    def __call__(self, sources=None, text=None, return_tensors=None, add_eos=True, **kwargs):
        """Turn humor sources into model inputs.

        Args:
            sources: a file path or list of paths (.srt, .txt, .md).
            text: a string or list of strings already extracted elsewhere
                (e.g. a book chapter); cleaned the same way as files.
            return_tensors: passed to the tokenizer ("pt" needs torch and padding).
            add_eos: end every document with an eos_token (§2.4).
            **kwargs: passed to the tokenizer (padding, truncation, max_length...).

        Returns:
            The tokenizer's BatchEncoding: input_ids and attention_mask,
            one entry per document.
        """
        if sources is None and text is None:
            raise ValueError("Provide 'sources' and/or 'text'")
        if isinstance(sources, (str, Path)):
            sources = [sources]
        if isinstance(text, str):
            text = [text]

        documents = [d for s in sources or [] for d in self.preprocess(s)]
        documents += [d for d in map(self.clean, text or []) if d]
        if add_eos:
            eos_token = self.manual_config.get('special_tokens').get('eos_token')
            documents = [d + eos_token for d in documents]
        kwargs.setdefault("verbose", False)  # specials exceed GPT-2's context; chunking comes later (§2.6)
        return self.tokenizer(documents, return_tensors=return_tensors, **kwargs)

    def decode(self, tokens: KasualDataset  ,*args, **kwargs):
        return self.tokenizer.decode(tokens.tokens.tolist(), *args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
