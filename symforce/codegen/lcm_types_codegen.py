# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import python_util
from symforce import typing as T


def lcm_symforce_types_data() -> T.Dict[str, T.Any]:
    """
    Returns data for template generation with ``lcm_templates/symforce_types.lcm.jinja``.
    """
    return dict(python_util=python_util, sf=sf)
