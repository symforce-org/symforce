"""
Internal initialization code. Should not be called by users.
"""

import contextlib

from symforce import logger
from symforce import typing as T


def modify_symbolic_api(sympy_module: T.Any) -> None:
    """
    Augment the sympy API for symforce use.

    Args:
        sympy_module (module):
    """
    override_symbol_new(sympy_module)
    override_simplify(sympy_module)
    override_limit(sympy_module)
    override_subs(sympy_module)
    add_scoping(sympy_module)
    add_custom_methods(sympy_module)
    override_solve(sympy_module)


def override_symbol_new(sympy_module: T.Any) -> None:
    """
    Override Symbol.__new__ to incorporate named scopes and default to reals instead of complex.

    Args:
        sympy_module (module):
    """
    if sympy_module.__package__ == "symengine":
        original_symbol = sympy_module.Symbol

        class Symbol(sympy_module.Symbol):  # type: ignore # mypy thinks sm.Symbol isn't defined
            def __init__(
                self, name: str, commutative: bool = True, real: bool = True, positive: bool = None
            ) -> None:
                scoped_name = ".".join(sympy_module.__scopes__ + [name])
                super().__init__(scoped_name, commutative=commutative, real=real, positive=positive)

                # TODO(hayk): This is not enabled, right now all symbols are commutative, real, but
                # not positive.
                # self.is_real = real
                # self.is_positive = positive
                # self.is_commutative = commutative

        sympy_module.Symbol = Symbol

        # Because we're creating a new subclass, we also need to override sm.symbols to use this one
        original_symbols = sympy_module.symbols

        def new_symbols(names: str, **args: T.Any) -> T.Union[T.Sequence[Symbol], Symbol]:
            cls = args.pop("cls", Symbol)
            return original_symbols(names, **dict(args, cls=cls))

        sympy_module.symbols = new_symbols

    else:
        assert sympy_module.__package__ == "sympy"

        # Save original
        original_symbol_new = sympy_module.Symbol.__new__

        @staticmethod  # type: ignore
        def new_symbol(
            cls: T.Any,
            name: str,
            commutative: bool = True,
            real: bool = True,
            positive: bool = None,
        ) -> None:
            name = ".".join(sympy_module.__scopes__ + [name])
            obj = original_symbol_new(
                cls, name, commutative=commutative, real=real, positive=positive
            )
            return obj

        sympy_module.Symbol.__new__ = new_symbol


def override_simplify(sympy_module: T.Type) -> None:
    """
    Override simplify so that we can use it with the symengine backend

    Args:
        sympy_module (module):
    """
    if hasattr(sympy_module, "simplify"):
        return

    import sympy

    def simplify(*args: T.Any, **kwargs: T.Any) -> sympy.S:
        logger.warning("Converting to sympy to use .simplify")
        return sympy_module.S(sympy.simplify(sympy.S(*args), **kwargs))

    sympy_module.simplify = simplify


def override_limit(sympy_module: T.Type) -> None:
    """
    Override limit so that we can use it with the symengine backend

    Args:
        sympy_module (module):
    """
    if hasattr(sympy_module, "limit"):
        return

    import sympy

    def limit(e: T.Any, z: T.Any, z0: T.Any, dir: str = "+") -> sympy.S:
        logger.warning("Converting to sympy to use .limit")
        return sympy_module.S(sympy.limit(sympy.S(e), sympy.S(z), sympy.S(z0), dir=dir))

    sympy_module.limit = limit


def create_named_scope(scopes_list: T.List[str]) -> T.Callable:
    """
    Return a context manager that adds to the given list of name scopes. This is used to
    add scopes to symbol names for namespacing.
    """

    @contextlib.contextmanager
    def named_scope(scope: str) -> T.Iterator[None]:
        scopes_list.append(scope)

        # The body of the with block is executed inside the yield, this ensures we release the
        # scope if something in the block throws
        try:
            yield None
        finally:
            scopes_list.pop()

    return named_scope


def add_scoping(sympy_module: T.Type) -> None:
    """
    Add name scopes to the symbolic API.
    """

    def set_scope(scope: str) -> None:
        sympy_module.__scopes__ = scope.split(".") if scope else []

    setattr(sympy_module, "set_scope", set_scope)
    setattr(sympy_module, "get_scope", lambda: ".".join(sympy_module.__scopes__))
    sympy_module.set_scope("")

    setattr(sympy_module, "scope", create_named_scope(sympy_module.__scopes__))


def _flatten_storage_type_subs(subs_dict: T.MutableMapping) -> None:
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


def _get_subs_dict(*args: T.Any) -> T.Dict:
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


def override_subs(sympy_module: T.Type) -> None:
    """
    Patch subs to support storage classes in substitution by calling to_storage() to flatten
    the substitution dict. This has to be done slightly differently in symengine and sympy.
    """
    if sympy_module.__name__ == "symengine":
        # For some reason this doesn't exist unless we import the symengine_wrapper directly as a
        # local variable, i.e. just `import symengine.lib.symengine_wrapper` does not let us access
        # symengine.lib.symengine_wrapper
        import symengine.lib.symengine_wrapper as wrapper

        original_get_dict = wrapper.get_dict
        wrapper.get_dict = lambda *args: original_get_dict(_get_subs_dict(*args))
    elif sympy_module.__name__ == "sympy":
        original_subs = sympy_module.Basic.subs
        sympy_module.Basic.subs = lambda self, *args, **kwargs: original_subs(
            self, _get_subs_dict(*args), **kwargs
        )
    else:
        raise NotImplementedError(f"Unknown backend: '{sympy_module.__name__}'")


def override_solve(sympy_module: T.Type) -> None:
    """
    Patch solve to make symengine's API consistent with SymPy's.  Currently this only supports
    solutions expressed by symengine as an sm.FiniteSet or EmptySet
    """
    if sympy_module.__name__ == "symengine":
        original_solve = sympy_module.solve

        # Unfortunately this already doesn't have a docstring or argument list in symengine
        def solve(*args: T.Any, **kwargs: T.Any) -> T.List[T.Scalar]:
            solution = original_solve(*args, **kwargs)
            from symengine.lib.symengine_wrapper import EmptySet

            if isinstance(solution, sympy_module.FiniteSet):
                return list(solution.args)
            elif isinstance(solution, EmptySet):
                return []
            else:
                raise NotImplementedError(
                    f"sm.solve currently only supports FiniteSet and EmptySet results on the SymEngine backend, got {type(solution)} instead"
                )

        sympy_module.solve = solve
    elif sympy_module.__name__ == "sympy":
        # This one is fine as is
        return
    else:
        raise NotImplementedError(f"Unknown backend: '{sympy_module.__name__}'")


def add_custom_methods(sympy_module: T.Type) -> None:
    """
    Add safe helper methods to the symbolic API.
    """

    def register(func: T.Callable) -> T.Callable:
        setattr(sympy_module, func.__name__, func)
        return func

    @register
    def atan2_safe(y: T.Scalar, x: T.Scalar, epsilon: T.Scalar = 0) -> T.Scalar:
        return sympy_module.atan2(y, x + (sympy_module.sign(x) + 0.5) * epsilon)

    @register
    def asin_safe(x: T.Scalar, epsilon: T.Scalar = 0) -> T.Scalar:
        x_safe = sympy_module.Max(-1 + epsilon, sympy_module.Min(1 - epsilon, x))
        return sympy_module.asin(x_safe)

    @register
    def acos_safe(x: T.Scalar, epsilon: T.Scalar = 0) -> T.Scalar:
        x_safe = sympy_module.Max(-1 + epsilon, sympy_module.Min(1 - epsilon, x))
        return sympy_module.acos(x_safe)

    @register
    def sign_no_zero(x: T.Scalar) -> T.Scalar:
        """
        Returns -1 if x is negative, 1 if x is positive, and 1 if x is zero.
        """
        return 2 * sympy_module.Min(sympy_module.sign(x), 0) + 1

    @register
    def copysign_no_zero(x: T.Scalar, y: T.Scalar) -> T.Scalar:
        """
        Returns a value with the magnitude of x and sign of y. If y is zero, returns positive x.
        """
        return sympy_module.Abs(x) * sign_no_zero(y)

    @register
    def argmax_onehot(vals: T.Sequence[T.Scalar]) -> T.List[T.Scalar]:
        """
        Returns a list l such that l[i] = 1.0 if i is the smallest index such that
        vals[i] equals Max(*vals). l[i] = 0.0 otherwise.

        Precondition:
            vals has at least one element
        """
        m = sympy_module.Max(*vals)
        results = []
        have_max_already = 0
        for val in vals:
            results.append(
                sympy_module.logical_and(
                    sympy_module.is_nonnegative(val - m),
                    sympy_module.logical_not(have_max_already, unsafe=True),
                    unsafe=True,
                )
            )
            have_max_already = sympy_module.logical_or(have_max_already, results[-1], unsafe=True)
        return results

    @register
    def argmax(vals: T.Sequence[T.Scalar]) -> T.Scalar:
        """
        Returns i (as a T.Scalar) such that i is the smallest index such that
        vals[i] equals Max(*vals).

        Precondition:
            vals has at least one element
        """
        return sum([i * val for i, val in enumerate(argmax_onehot(vals))])

    from symforce import logic

    logic.add_logic_methods(sympy_module)
