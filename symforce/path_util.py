# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import json
import typing as T
from pathlib import Path


class MissingManifestException(RuntimeError):
    pass


class _Manifest:
    """
    Internal class to manage loading data from the build manifest and caching that data.  Not
    intended for use outside of path_util.py.
    """

    _manifest = None

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._manifest is None:
            manifest_path = Path(__file__).parent.parent / "build" / "manifest.json"
            try:
                with open(manifest_path) as f:
                    cls._manifest = json.load(f)
            except FileNotFoundError as ex:
                raise MissingManifestException(f"Manifest not found at {manifest_path}") from ex

    @classmethod
    def get_entry(cls, key: str) -> Path:
        cls._ensure_loaded()
        assert cls._manifest is not None
        return Path(cls._manifest[key]).resolve()

    @classmethod
    def get_entries(cls, key: str) -> T.List[Path]:
        cls._ensure_loaded()
        assert cls._manifest is not None
        return [Path(s).resolve() for s in cls._manifest[key]]


def symforce_dir() -> Path:
    return Path(__file__).parent.parent


def symenginepy_install_dir() -> Path:
    return _Manifest.get_entry("symenginepy_install_dir")


def cc_sym_install_dir() -> Path:
    return _Manifest.get_entry("cc_sym_install_dir")


def binary_output_dir() -> Path:
    return _Manifest.get_entry("binary_output_dir")
