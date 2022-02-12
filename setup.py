# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="symforce",
    version="0.2dev",
    author="Skydio, Inc",
    author_email="hayk@skydio.com",
    packages=find_packages(),
    license="Apache 2.0",
    long_description=Path("README.md").read_text(),
)
