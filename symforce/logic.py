"""
Internal initialization code for logic methods.  Should not be called by users

TODO(aaron): Make these methods show up in the API docs?
"""
from symforce import typing as T


def add_logic_methods(sympy_module: T.Type) -> None:
    """
    Add logical helper methods to the symbolic API
    """

    def register(func: T.Callable) -> T.Callable:
        setattr(sympy_module, func.__name__, func)
        return func

    @register
    def is_positive(x: T.Scalar) -> T.Scalar:
        """
        Returns 1 if x is positive, 0 otherwise
        """
        return sympy_module.Max(sympy_module.sign(x), 0)

    @register
    def is_negative(x: T.Scalar) -> T.Scalar:
        """
        Returns 1 if x is negative, 0 otherwise
        """
        return sympy_module.Max(sympy_module.sign(-x), 0)

    @register
    def is_nonnegative(x: T.Scalar) -> T.Scalar:
        """
        Returns 1 if x is >= 0, 0 if x is negative
        """
        return 1 - sympy_module.Max(sympy_module.sign(-x), 0)

    @register
    def is_nonpositive(x: T.Scalar) -> T.Scalar:
        """
        Returns 1 if x is <= 0, 0 if x is positive
        """
        return 1 - sympy_module.Max(sympy_module.sign(x), 0)

    @register
    def logical_and(a: T.Scalar, b: T.Scalar, unsafe: bool = False) -> T.Scalar:
        """
        Logical and of two Scalars

        Input values are treated as true if they are positive, false if they are 0 or negative.
        The returned value is 1 for true, 0 for false.

        If unsafe is True, the resulting expression is fewer ops but assumes the inputs are exactly
        0 or 1; results for other (finite) inputs will be finite, but are otherwise undefined.
        """
        if unsafe:
            return sympy_module.Min(a, b)
        else:
            return sympy_module.Max(sympy_module.sign(a) + sympy_module.sign(b), 1) - 1

    @register
    def logical_or(a: T.Scalar, b: T.Scalar, unsafe: bool = False) -> T.Scalar:
        """
        Logical or of two Scalars

        Input values are treated as true if they are positive, false if they are 0 or negative.
        The returned value is 1 for true, 0 for false.

        If unsafe is True, the resulting expression is fewer ops but assumes the inputs are exactly
        0 or 1; results for other (finite) inputs will be finite, but are otherwise undefined.
        """
        if unsafe:
            return sympy_module.Max(a, b)
        else:
            return sympy_module.Max(sympy_module.sign(a), sympy_module.sign(b), 0)

    @register
    def logical_not(a: T.Scalar, unsafe: bool = False) -> T.Scalar:
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
            return 1 - sympy_module.Max(sympy_module.sign(a), 0)
