"""Repository path helpers for optional external dependencies."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
THIRD_PARTY_ROOT = Path(os.environ.get("VBGS_THIRD_PARTY_DIR", REPO_ROOT / "third_party"))


def external_dir(name: str, env_var: str) -> Path:
    """Return external dependency path, preferring env override then third_party."""
    if override := os.environ.get(env_var):
        return Path(override).expanduser().resolve()

    preferred = THIRD_PARTY_ROOT / name
    if preferred.exists():
        return preferred.resolve()

    # Legacy fallback for old checkouts before external repos moved out of src/.
    legacy = REPO_ROOT / "src" / name
    return legacy.resolve()


def add_external_to_syspath(name: str, env_var: str) -> Path:
    path = external_dir(name, env_var)
    if str(path) not in sys.path:
        sys.path.append(str(path))
    return path


def gaussian_splatting_path() -> Path:
    return external_dir("gaussian-splatting", "VBGS_GAUSSIAN_SPLATTING_DIR")


def add_gaussian_splatting_to_syspath() -> Path:
    return add_external_to_syspath("gaussian-splatting", "VBGS_GAUSSIAN_SPLATTING_DIR")
