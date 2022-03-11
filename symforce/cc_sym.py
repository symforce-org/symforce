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

try:
    # If cc_sym is availble on the python path, use it
    from cc_sym import *  # pylint: disable=wildcard-import
except ImportError:
    sys.path.append(os.fspath(path_util.cc_sym_install_dir()))

    from cc_sym import *  # pylint: disable=wildcard-import
