# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Script to download Bundle-Adjustment-in-the-Large datasets into the `./data` folder
"""

import bz2
from pathlib import Path

import requests

CURRENT_DIR = Path(__file__).resolve().parent

URL = "https://grail.cs.washington.edu/projects/bal/data/trafalgar/"

PROBLEMS = [
    "problem-21-11315-pre.txt",
    "problem-39-18060-pre.txt",
    "problem-50-20431-pre.txt",
    "problem-126-40037-pre.txt",
    "problem-138-44033-pre.txt",
    "problem-161-48126-pre.txt",
    "problem-170-49267-pre.txt",
    "problem-174-50489-pre.txt",
    "problem-193-53101-pre.txt",
    "problem-201-54427-pre.txt",
    "problem-206-54562-pre.txt",
    "problem-215-55910-pre.txt",
    "problem-225-57665-pre.txt",
    "problem-257-65132-pre.txt",
]


def main() -> None:
    (CURRENT_DIR / "data").mkdir(exist_ok=True)
    for problem in PROBLEMS:
        result = requests.get(URL + problem + ".bz2")
        result_uncompressed = bz2.decompress(result.content)
        (CURRENT_DIR / "data" / problem).write_bytes(result_uncompressed)


if __name__ == "__main__":
    main()
