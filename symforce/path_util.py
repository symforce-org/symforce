import json
from pathlib import Path
import typing as T


class _Manifest:
    """
    Internal class to manage loading data from the build manifest and caching that data.  Not
    intended for use outside of path_util.py.
    """

    _manifest = None

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._manifest is None:
            with open(Path(__file__).parent.parent / "build" / "manifest.json") as f:
                cls._manifest = json.load(f)

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


def eigen_include_dirs() -> T.List[Path]:
    return _Manifest.get_entries("eigen_include_dirs")


def lcm_include_dirs() -> T.List[Path]:
    return _Manifest.get_entries("lcm_include_dirs")


def spdlog_include_dirs() -> T.List[Path]:
    return _Manifest.get_entries("spdlog_include_dirs")


def symforce_dir() -> Path:
    return Path(__file__).parent.parent


def catch2_include_dirs() -> T.List[Path]:
    return _Manifest.get_entries("catch2_include_dirs")


def lcm_gen_exe() -> Path:
    return _Manifest.get_entry("lcm_gen_exe")


def symenginepy_install_dir() -> Path:
    return _Manifest.get_entry("symenginepy_install_dir")
