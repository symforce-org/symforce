"""
Internal initialization code. Should not be called by users.
"""

import contextlib

from symforce import logger
from symforce import types as T


def modify_symbolic_api(sympy_module):
    # type: (T.Type) -> None
    """
    Augment the sympy API for symforce use.

    Args:
        sympy_module (module):
    """
    override_symbol_new(sympy_module)
    override_simplify(sympy_module)
    override_subs(sympy_module)
    add_scoping(sympy_module)
    add_custom_methods(sympy_module)


def override_symbol_new(sympy_module):
    # type: (T.Type) -> None
    """
    Override Symbol.__new__ to incorporate named scopes and default to reals instead of complex.

    Args:
        sympy_module (module):
    """
    if sympy_module.__package__ == "symengine":
        original_symbol_init = sympy_module.Symbol.__init__

        def init_symbol(self, name, commutative=True, real=True, positive=None):
            # type: (T.Any, str, bool, bool, bool) -> None
            scoped_name = ".".join(sympy_module.__scopes__ + [name])
            original_symbol_init(
                self, scoped_name, commutative=commutative, real=real, positive=positive
            )

            # TODO(hayk): This is not enabled, right now all symbols are commutative, real, but not positive.
            # self.is_real = real
            # self.is_positive = positive
            # self.is_commutative = commutative

        sympy_module.Symbol.__init__ = init_symbol
    else:
        assert sympy_module.__package__ == "sympy"

        # Save original
        original_symbol_new = sympy_module.Symbol.__new__

        @staticmethod  # type: ignore
        def new_symbol(cls, name, commutative=True, real=True, positive=None):
            # type: (T.Any, str, bool, bool, bool) -> None
            name = ".".join(sympy_module.__scopes__ + [name])
            obj = original_symbol_new(
                cls, name, commutative=commutative, real=real, positive=positive
            )
            return obj

        sympy_module.Symbol.__new__ = new_symbol


def override_simplify(sympy_module):
    # type: (T.Type) -> None
    """
    Override simplify so that we can use it with the symengine backend

    Args:
        sympy_module (module):
    """
    if hasattr(sympy_module, "simplify"):
        return

    import sympy

    def simplify(*args, **kwargs):
        # type: (T.Any, T.Any) -> sympy.S
        logger.warning("Converting to sympy to use .simplify")
        return sympy.S(sympy.simplify(sympy.S(*args), **kwargs))

    sympy_module.simplify = simplify


def create_named_scope(scopes_list):
    # type: (T.List[str]) -> T.Callable
    """
    Return a context manager that adds to the given list of name scopes. This is used to
    add scopes to symbol names for namespacing.
    """

    @contextlib.contextmanager
    def named_scope(scope):
        # type: (str) -> T.Iterator[None]
        scopes_list.append(scope)
        yield None
        scopes_list.pop()

    return named_scope


def add_scoping(sympy_module):
    # type: (T.Type) -> None
    """
    Add name scopes to the symbolic API.
    """

    def set_scope(scope):
        # type: (str) -> None
        sympy_module.__scopes__ = scope.split(".") if scope else []

    setattr(sympy_module, "set_scope", set_scope)
    setattr(sympy_module, "get_scope", lambda: ".".join(sympy_module.__scopes__))
    sympy_module.set_scope("")

    setattr(sympy_module, "scope", create_named_scope(sympy_module.__scopes__))


def _flatten_storage_type_subs(subs_dict):
    # type: (T.MutableMapping) -> None
    """
    Replace storage types with their scalar counterparts
    """
    keys = list(subs_dict.keys())
    for key in keys:
        if hasattr(key, "to_storage"):
            new_keys = key.to_storage()
            assert type(key) == type(subs_dict[key])
            new_values = subs_dict[key].to_storage()
            for i, new_key in enumerate(new_keys):
                subs_dict[new_key] = new_values[i]
            del subs_dict[key]


def _get_subs_dict(*args):
    # type: (T.Any) -> T.Dict
    """
    Handle args to subs being a single key-value pair or a dict.
    """
    if len(args) == 2:
        subs_dict = {args[0]: args[1]}
    elif len(args) == 1:
        subs_dict = dict(args[0])

    assert isinstance(subs_dict, T.Mapping)
    _flatten_storage_type_subs(subs_dict)

    return subs_dict


def override_subs(sympy_module):
    # type: (T.Type) -> None
    """
    Patch subs to support storage classes in substitution by calling to_storage() to flatten
    the substitution dict. This has to be done slightly differently in symengine and sympy.
    """
    if sympy_module.__name__ == "symengine":
        original_get_dict = sympy_module.lib.symengine_wrapper.get_dict
        sympy_module.lib.symengine_wrapper.get_dict = lambda *args: original_get_dict(
            _get_subs_dict(*args)
        )
    elif sympy_module.__name__ == "sympy":
        original_subs = sympy_module.Basic.subs
        sympy_module.Basic.subs = lambda self, *args, **kwargs: original_subs(
            self, _get_subs_dict(*args), **kwargs
        )
    else:
        raise NotImplementedError("Unknown backend: '{}'".format(sympy_module.__name__))


def add_custom_methods(sympy_module):
    # type: (T.Type) -> None
    """
    Add safe helper methods to the symbolic API.
    """

    def atan2_safe(y, x, epsilon=0):
        # type: (T.Scalar, T.Scalar, T.Scalar) -> T.Scalar
        return sympy_module.atan2(y, x + (sympy_module.sign(x) + 0.5) * epsilon)

    setattr(sympy_module, "atan2_safe", atan2_safe)

    def asin_safe(x, epsilon=0):
        # type: (T.Scalar, T.Scalar) -> T.Scalar
        # TODO (nathan): Consider using asin(max(-1, min(1, x))) in the future
        return sympy_module.asin(x - sympy_module.sign(x) * epsilon)

    setattr(sympy_module, "asin_safe", asin_safe)

    def sign_no_zero(x, epsilon=0):
        # type: (T.Scalar, T.Scalar) -> T.Scalar
        """
        Returns -1 if x is negative, 1 if x is positive, and 1 if x is zero (given a positive epsilon).
        """
        return sympy_module.sign(x + (sympy_module.sign(x) + sympy_module.S(1) / 2) * epsilon)

    setattr(sympy_module, "sign_no_zero", sign_no_zero)

    def copysign_no_zero(x, y, epsilon=0):
        # type: (T.Scalar, T.Scalar, T.Scalar) -> T.Scalar
        """
        Returns a value with the magnitude of x and sign of y. If y is zero, returns positive x.
        """
        return sympy_module.Abs(x) * sign_no_zero(y, epsilon)

    setattr(sympy_module, "copysign_no_zero", copysign_no_zero)
