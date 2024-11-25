# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Script to download Bundle-Adjustment-in-the-Large datasets into the `./data` folder
"""

from __future__ import annotations

import bz2
import re
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

import argh
import requests

CURRENT_DIR = Path(__file__).resolve().parent

BASE_URL = "https://grail.cs.washington.edu/projects/bal"


class BALParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for attr, maybe_value in attrs:
                if attr == "href" and maybe_value is not None:
                    self.links.append(maybe_value)


DATASETS = ["ladybug", "trafalgar", "dubrovnik", "venice", "final"]


@argh.arg("dataset", choices=DATASETS)
def main(dataset: str) -> None:
    parser = BALParser()
    parser.feed(requests.get(f"{BASE_URL}/{dataset}.html").content.decode())

    (CURRENT_DIR / "data" / dataset).mkdir(exist_ok=True)
    for data_url in parser.links:
        if not re.match(rf"data/{dataset}/problem-.+\.txt\.bz2", data_url):
            continue

        print(f"Downloading {data_url}")

        result = requests.get(f"{BASE_URL}/{data_url}")
        result_uncompressed = bz2.decompress(result.content)
        problem_name = Path(urllib.parse.urlparse(data_url).path).stem
        (CURRENT_DIR / "data" / dataset / problem_name).write_bytes(result_uncompressed)


if __name__ == "__main__":
    argh.dispatch_command(main)
