#!/usr/bin/env python3
"""Check that a public GRL-DEACO release does not track private artifacts."""

import subprocess
import sys
from pathlib import Path


BLOCKED_SUFFIXES = {
    ".glb",
    ".obj",
    ".stl",
    ".pth",
    ".pt",
    ".ckpt",
    ".log",
}

BLOCKED_PARTS = {
    "static/glb",
    "datasets",
    "data",
    "scenarios_rl_dataset",
    "rl_training",
    "results",
    "evaluation_results",
    "checkpoints",
    "trained_models",
}


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def is_blocked(path: Path) -> bool:
    normalized = path.as_posix()
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return True
    return any(part in normalized for part in BLOCKED_PARTS)


def main() -> int:
    blocked = [path for path in tracked_files() if is_blocked(path)]
    if blocked:
        print("Blocked private/generated files are tracked:")
        for path in blocked:
            print(f"  - {path}")
        return 1

    required = [
        "README.md",
        "CITATION.cff",
        "configs/paper_reproduction_config.yaml",
        "docs/REPRODUCIBILITY.md",
        "docs/DATA_FORMAT.md",
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        print("Required release files are missing:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("Public release check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
