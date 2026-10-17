"""Strict, self-contained parsing for RTCFake challenge protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

LABEL_TO_ID = {"spoof": 0, "bonafide": 1}
ALLOWED_SPLITS = {"train", "dev", "progress", "eval"}
ALLOWED_VIEWS = {"offline", "online"}
ALLOWED_LANGUAGES = {"en", "zh"}


@dataclass(frozen=True)
class Example:
    utterance_id: str
    split: str
    view: str
    language: str
    label: int | None


def validate_utterance_id(value: str, *, split: str) -> tuple[str, str]:
    """Reject absolute/traversal paths and enforce the known dataset layout."""
    if "\\" in value or value.startswith("/") or ":" in value:
        raise ValueError(f"Unsafe audio path: {value}")
    path = PurePosixPath(value)
    if len(path.parts) != 3 or any(p in {"", ".", ".."} for p in path.parts):
        raise ValueError(f"Expected view/language/filename.wav: {value}")
    view, lang, name = path.parts
    if view not in ALLOWED_VIEWS or lang not in ALLOWED_LANGUAGES:
        raise ValueError(f"Unsupported view or language in {value}")
    if split in {"progress", "eval"} and view != "online":
        raise ValueError("Held-out protocols must contain online audio only")
    if not name.lower().endswith(".wav"):
        raise ValueError(f"Not a WAV path: {value}")
    return view, lang


def read_protocol(file_path: str | Path, split: str) -> list[Example]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Invalid split: {split}")
    expected_fields = 2 if split in {"train", "dev"} else 1
    seen: set[str] = set()
    examples = []
    with Path(file_path).open("r", encoding="utf-8") as source:
        for line_no, raw in enumerate(source, 1):
            parts = raw.strip().split()
            if not parts:
                continue
            if len(parts) != expected_fields:
                raise ValueError(
                    f"{file_path}:{line_no}: expected {expected_fields} fields, got {len(parts)}"
                )
            uid = parts[0]
            view, lang = validate_utterance_id(uid, split=split)
            if uid in seen:
                raise ValueError(f"Duplicate utterance ID at line {line_no}: {uid}")
            seen.add(uid)
            label = None
            if expected_fields == 2:
                label_name = parts[1].lower()
                if label_name not in LABEL_TO_ID:
                    raise ValueError(f"Invalid class at line {line_no}: {parts[1]}")
                label = LABEL_TO_ID[label_name]
            examples.append(Example(uid, split, view, lang, label))
    if not examples:
        raise ValueError(f"Empty protocol: {file_path}")
    return examples
