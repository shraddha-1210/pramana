"""Fail if an unresolved citation marker reaches a commit.

Build plan Part 0, Rule 2: a source that is not in ``docs/references.md`` may not be
cited. Where one is needed, the author leaves a marker and stops, rather than filling
in a plausible author and year. This hook is the cheap mechanical enforcement of that
rule -- it makes the marker impossible to forget rather than merely discouraged.

Usage (pre-commit passes staged paths):

    python scripts/check_citations.py [path ...]

Exit code 0 if no marker is found, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Assembled rather than written literally so this file does not trip its own check.
MARKER = "CITATION" + " NEEDED"

# Files that legitimately discuss the marker rather than carrying an unresolved one.
SELF_REFERENTIAL = {
    Path("scripts/check_citations.py"),
    Path("docs/references.md"),
    Path("docs/plan.md"),
}


def scan(paths: list[Path]) -> list[tuple[Path, int, str]]:
    """Return (path, 1-based line number, line) for every marker found."""
    hits: list[tuple[Path, int, str]] = []
    for path in paths:
        if Path(*path.parts) in SELF_REFERENTIAL or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: nothing to cite in it
        for lineno, line in enumerate(text.splitlines(), start=1):
            if MARKER in line:
                hits.append((path, lineno, line.strip()))
    return hits


def main(argv: list[str]) -> int:
    """Report every marker found in ``argv``. Returns 1 if any was found."""
    hits = scan([Path(a) for a in argv])
    if not hits:
        return 0
    print(f"Unresolved {MARKER} marker(s) -- resolve or remove before committing:\n")
    for path, lineno, line in hits:
        print(f"  {path}:{lineno}: {line}")
    print(
        "\nAdd the source to docs/references.md only if a team member has opened it "
        "(build plan Rule 2). Do not substitute a plausible reference."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
