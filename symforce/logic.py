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


def less_equal(x: sf.Scalar, y: sf.Scalar) -> sf.Scalar:
    """
    Returns 1 if x <= y, 0 otherwise
    """
    return is_nonpositive(x - y)


def greater_equal(x: sf.Scalar, y: sf.Scalar) -> sf.Scalar:
    """
    Returns 1 if x >= y, 0 otherwise
    """
    return is_nonnegative(x - y)


def less(x: sf.Scalar, y: sf.Scalar) -> sf.Scalar:
    """
    Returns 1 if x < y, 0 otherwise
    """
    return is_negative(x - y)


def greater(x: sf.Scalar, y: sf.Scalar) -> sf.Scalar:
    """
    Returns 1 if x > y, 0 otherwise
    """
    return is_positive(x - y)


def logical_and(*args: sf.Scalar, unsafe: bool = False) -> sf.Scalar:
    """
    Logical and of two or more Scalars

    Input values are treated as true if they are positive, false if they are 0 or negative.
    The returned value is 1 for true, 0 for false.

    If unsafe is True, the resulting expression is fewer ops but assumes the inputs are exactly
    0 or 1; results for other (finite) inputs will be finite, but are otherwise undefined.
    """
    if unsafe:
        return sf.Min(*args)
    else:
        return sf.Max(sum(sf.sign(x) for x in args), len(args) - 1) - (len(args) - 1)


def logical_or(*args: sf.Scalar, unsafe: bool = False) -> sf.Scalar:
    """
    Logical or of two or more Scalars

    Input values are treated as true if they are positive, false if they are 0 or negative.
    The returned value is 1 for true, 0 for false.

    If unsafe is True, the resulting expression is fewer ops but assumes the inputs are exactly
    0 or 1; results for other (finite) inputs will be finite, but are otherwise undefined.
    """
    if unsafe:
        return sf.Max(*args)
    else:
        return sf.Max(*[sf.sign(x) for x in args], 0)


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
