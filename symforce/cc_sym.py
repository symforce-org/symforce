# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Locate and re-export cc_sym as symforce.cc_sym
"""
import logging as _logging
import os
import sys

from symforce import logger as _logger
from symforce import path_util

try:
    # If cc_sym is availble on the python path, use it
    from cc_sym import *  # pylint: disable=wildcard-import
except ImportError as ex:
    try:
        sys.path.append(os.fspath(path_util.cc_sym_install_dir()))
    except path_util.MissingManifestException as ex2:
        raise ImportError from ex2

    from cc_sym import *  # pylint: disable=wildcard-import

# Set log level, in case the user has set the level in python
set_log_level(  # pylint: disable=undefined-variable
    _logging.getLevelName(_logger.getEffectiveLevel())
)
