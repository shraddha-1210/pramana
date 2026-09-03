"""Shared pytest configuration.

``scripts/`` is put on the import path so ``tests/test_env.py`` can import
``verify_env`` directly. It is a script rather than a package module by design:
it must be runnable standalone, before the package is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
