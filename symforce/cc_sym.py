# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Locate and re-export cc_sym as symforce.cc_sym
"""
import os
import sys

from symforce import path_util

sys.path.append(os.fspath(path_util.cc_sym_install_dir()))

from cc_sym import *  # pylint: disable=wildcard-import
