from pathlib import Path

***REMOVED*** = ***REMOVED***


def eigen_include_dir() -> Path:
    return (***REMOVED*** / "include" / "eigen3").resolve()


def lcm_include_dir() -> Path:
    return (***REMOVED*** / "include").resolve()


def spdlog_include_dir() -> Path:
    return (***REMOVED*** / "_deps" / "spdlog-src" / "include").resolve()


def symforce_dir() -> Path:
    return Path(__file__).parent.parent


def catch2_include_dir() -> Path:
    return symforce_dir() / "third_party" / "catch2"


def fmt_library_dir() -> Path:
    return (***REMOVED*** / "lib").resolve()


def spdlog_library_dir() -> Path:
    return (***REMOVED*** / "lib").resolve()


def lcm_gen_exe() -> Path:
    return (***REMOVED*** / "bin" / "lcm-gen").resolve()


def skymarshal_exe() -> Path:
    return ***REMOVED*** / "bin" / "skymarshal"
