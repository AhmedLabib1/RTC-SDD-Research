"""Strict, deterministic conversion of pinned training requirements."""

from __future__ import annotations

import re
from pathlib import Path

FAIRSEQ_SHA = "a54021305d6b3c4c5959ac9395135f63202db8f1"

TORCH_INDEX = "https://download.pytorch.org/whl/cu121"

REQUIRED_PINS = (
    "torch==2.5.1+cu121",
    "torchaudio==2.5.1+cu121",
    "torchvision==0.20.1+cu121",
)

FAIRSEQ_LEGACY = re.compile(
    r"-e\s+git\+https://github\.com/pytorch/fairseq"
    r"(?:\.git)?@([0-9a-fA-F]{40})#egg=fairseq"
)


def convert_requirements(source: str) -> str:
    """Convert only the pinned Fairseq editable VCS reference.

    Every other requirement remains unchanged.
    """

    lines = []
    converted = 0

    for original in source.splitlines():
        line = original.strip()

        match = FAIRSEQ_LEGACY.fullmatch(line)

        if match:
            sha = match.group(1).lower()

            if sha != FAIRSEQ_SHA:
                raise ValueError(
                    "Fairseq commit differs from the approved pin"
                )

            lines.append(
                "fairseq @ "
                "git+https://github.com/pytorch/fairseq.git@"
                + sha
            )
            converted += 1

        else:
            if line.startswith("-e git+"):
                raise ValueError(
                    "Unknown editable Git requirement"
                )

            lines.append(original)

    if converted != 1:
        raise ValueError(
            f"Expected exactly one pinned Fairseq reference, "
            f"found {converted}"
        )

    stripped = [line.strip() for line in lines]

    for requirement in REQUIRED_PINS:
        if stripped.count(requirement) != 1:
            raise ValueError(
                f"Missing or duplicated pin: {requirement}"
            )

    return "\n".join(lines) + "\n"


def install_commands(
    *,
    python: Path,
    requirements: Path,
    project: Path,
) -> list[list[str]]:
    """Generate explicit installation commands; execute nothing."""

    python = Path(python).absolute()
    requirements = Path(requirements).absolute()
    project = Path(project).absolute()

    return [
        [
            "uv", "pip", "install",
            "--python", str(python),
            "--default-index",
            "https://pypi.org/simple",
            "--extra-index-url", TORCH_INDEX,
            "--index-strategy", "unsafe-first-match",
            "-r", str(requirements),
        ],
        [
            "uv", "pip", "install",
            "--python", str(python),
            "-e", str(project),
            "huggingface_hub>=0.30,<1.0",
        ],
        [
            "uv", "pip", "check",
            "--python", str(python),
        ],
    ]
