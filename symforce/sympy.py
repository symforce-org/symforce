# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
DEPRECATED

The old home of the SymForce-modified SymPy or SymEngine module.  This is deprecated, please use
`symforce.symbolic` instead.
"""

# TODO(aaron): Remove this module once present in a release or two
import warnings

warnings.warn("`symforce.sympy` is deprecated, use `symforce.symbolic`", FutureWarning)

from symforce.internal.symbolic import *  # pylint: disable=wildcard-import,unused-wildcard-import
