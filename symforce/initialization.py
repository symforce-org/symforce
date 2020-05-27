"""
Internal initialization code. Should not be called by users.
"""

import contextlib

from symforce import logger


def modify_symbolic_api(sympy_module):
    """
    Augment the sympy API for symforce use.

    Args:
        sympy_module (module):
    """
    override_symbol_new(sympy_module)
    override_simplify(sympy_module)
    add_scoping(sympy_module)
    add_safe_methods(sympy_module)


def override_symbol_new(sympy_module):
    """
    Override Symbol.__new__ to incorporate named scopes and default to reals instead of complex.

    Args:
        sympy_module (module):
    """
    if sympy_module.__package__ == "symengine":
        original_symbol_init = sympy_module.Symbol.__init__

        def init_symbol(self, name, commutative=True, real=True, positive=None):
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

        @staticmethod
        def new_symbol(cls, name, commutative=True, real=True, positive=None):
            name = ".".join(sympy_module.__scopes__ + [name])
            obj = original_symbol_new(
                cls, name, commutative=commutative, real=real, positive=positive
            )
            return obj

        sympy_module.Symbol.__new__ = new_symbol


def override_simplify(sympy_module):
    """
    Override simplify so that we can use it with the symengine backend

    Args:
        sympy_module (module):
    """
    if hasattr(sympy_module, "simplify"):
        return

    import sympy

    def simplify(*args, **kwargs):
        logger.warning("Converting to sympy to use .simplify")
        return sympy.S(sympy.simplify(sympy.S(*args), **kwargs))

    sympy_module.simplify = simplify


def create_named_scope(scopes_list):
    """
    Return a context manager that adds to the given list of name scopes. This is used to
    add scopes to symbol names for namespacing.

    Args:
        scopes_list (list(str)):

    Returns:
        contextmanager:
    """

    @contextlib.contextmanager
    def named_scope(scope):
        scopes_list.append(scope)
        yield None
        scopes_list.pop()

    return named_scope


def add_scoping(sympy_module):
    """
    Add name scopes to the symbolic API.

    Args:
        sympy_module (module):
    """

    def set_scope(scope):
        sympy_module.__scopes__ = scope.split(".") if scope else []

    setattr(sympy_module, "set_scope", set_scope)
    setattr(sympy_module, "get_scope", lambda: ".".join(sympy_module.__scopes__))
    sympy_module.set_scope("")

    setattr(sympy_module, "scope", create_named_scope(sympy_module.__scopes__))


def add_safe_methods(sympy_module):
    """
    Add safe helper methods to the symbolic API.

    Args:
        sympy_module (module):
    """

    def atan2_safe(y, x, epsilon=0):
        return sympy_module.atan2(y, x + (sympy_module.sign(x) + 0.5) * epsilon)

    setattr(sympy_module, "atan2_safe", atan2_safe)
