# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Functions for dealing with logical operations represented by scalars
"""

import symforce.internal.symbolic as sf


def is_positive(x: sf.Scalar) -> sf.Scalar:
    """
    Returns 1 if x is positive, 0 otherwise
    """
    return sf.Max(sf.sign(x), 0)


def is_negative(x: sf.Scalar) -> sf.Scalar:
    """
    Returns 1 if x is negative, 0 otherwise
    """
    return sf.Max(sf.sign(-x), 0)


def is_nonnegative(x: sf.Scalar) -> sf.Scalar:
    """
    Returns 1 if x is >= 0, 0 if x is negative
    """
    return 1 - sf.Max(sf.sign(-x), 0)


def is_nonpositive(x: sf.Scalar) -> sf.Scalar:
    """
    Returns 1 if x is <= 0, 0 if x is positive
    """
    return 1 - sf.Max(sf.sign(x), 0)


def logical_and(a: sf.Scalar, b: sf.Scalar, unsafe: bool = False) -> sf.Scalar:
    """
    Logical and of two Scalars

    Input values are treated as true if they are positive, false if they are 0 or negative.
    The returned value is 1 for true, 0 for false.

    If unsafe is True, the resulting expression is fewer ops but assumes the inputs are exactly
    0 or 1; results for other (finite) inputs will be finite, but are otherwise undefined.
    """
    if unsafe:
        return sf.Min(a, b)
    else:
        return sf.Max(sf.sign(a) + sf.sign(b), 1) - 1


def logical_or(a: sf.Scalar, b: sf.Scalar, unsafe: bool = False) -> sf.Scalar:
    """
    Logical or of two Scalars

    Input values are treated as true if they are positive, false if they are 0 or negative.
    The returned value is 1 for true, 0 for false.

    If unsafe is True, the resulting expression is fewer ops but assumes the inputs are exactly
    0 or 1; results for other (finite) inputs will be finite, but are otherwise undefined.
    """
    if unsafe:
        return sf.Max(a, b)
    else:
        return sf.Max(sf.sign(a), sf.sign(b), 0)


def logical_not(a: sf.Scalar, unsafe: bool = False) -> sf.Scalar:
    """
    Logical not of a Scalar

    Input value is treated as true if it is positive, false if it is 0 or negative. The
    returned value is 1 for true, 0 for false.

    If unsafe is True, the resulting expression is fewer ops but assumes the inputs are exactly
    0 or 1; results for other (finite) inputs will be finite, but are otherwise undefined.
    """
    if unsafe:
        return 1 - a
    else:
        return 1 - sf.Max(sf.sign(a), 0)
