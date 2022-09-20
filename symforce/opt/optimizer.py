# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from lcmtypes.sym._index_entry_t import index_entry_t
from lcmtypes.sym._optimization_iteration_t import optimization_iteration_t
from lcmtypes.sym._optimizer_params_t import optimizer_params_t
from lcmtypes.sym._values_t import values_t

from symforce import cc_sym
from symforce import typing as T
from symforce.opt.factor import Factor
from symforce.opt.numeric_factor import NumericFactor
from symforce.values import Values


class Optimizer:
    """
    A nonlinear least-squares optimizer

    Typical usage is to construct an Optimizer from a set of factors and keys to optimize, and then
    call `optimize` repeatedly with a `Values`.

    Example creation with an `OptimizationProblem` using `make_numeric_factors()`. The linearization
    functions are generated in `make_numeric_factors()` and are linearized with respect to
    `problem.optimized_keys()`.

        problem = OptimizationProblem(subproblems=[...], residual_blocks=...)
        factors = problem.make_numeric_factors("my_problem")
        optimizer = Optimizer(factors)

    Example creation with an `OptimizationProblem` using `make_symbolic_factors()`. The symbolic
    factors are converted into numeric factors when the optimizer is created, and are linearized
    with respect to the "optimized keys" passed to the optimizer. The linearization functions
    are generated when converting to numeric factors when the optimizer is created.

        problem = OptimizationProblem(subproblems=[...], residual_blocks=...)
        factors = problem.make_symbolic_factors("my_problem")
        optimizer = Optimizer(factors, problem.optimized_keys())

    Example creation with a single `Factor`:

        factor = Factor(
            [my_key_0, my_key_1, my_key_2], my_residual_function
        )
        optimizer = Optimizer(
            factors=[factor],
            optimized_keys=[my_key_0, my_key_1],
        )

    And usage:

        initial_guess = Values(...)
        result = optimizer.optimize(initial_guess)
        print(result.optimized_values)

    Wraps the C++ `sym::Optimizer` class in `opt/optimizer.h`, so the API is mostly the same and
    optimization results will be identical.

    Args:
        factors: A sequence of either Factor or NumericFactor objects representing the
            residuals in the problem. If (symbolic) Factors are passed, they are convered to
            NumericFactors by generating linearization functions of the residual with respect to the
            keys in `optimized_keys`.
        optimized_keys: A set of the keys to be optimized. Only required if symbolic factors are
            passed to the optimizer.
        params: Params for the optimizer
        debug_stats: Whether the optimizer should record debuggins stats such as the optimized
            values, residual, jacobian, etc. computed at each iteration of the optimization.
    """

    @dataclass
    class Params:
        """
        Parameters for the Python Optimizer

        Mirrors the optimizer_params_t LCM type, see documentation there for information on each
        parameter
        """

        verbose: bool = True
        initial_lambda: float = 1.0
        lambda_up_factor: float = 4.0
        lambda_down_factor: float = 1 / 4.0
        lambda_lower_bound: float = 0.0
        lambda_upper_bound: float = 1000000.0
        use_diagonal_damping: bool = False
        use_unit_damping: bool = True
        keep_max_diagonal_damping: bool = False
        diagonal_damping_min: float = 1e-6
        iterations: int = 50
        early_exit_min_reduction: float = 1e-6
        enable_bold_updates: bool = False

    @dataclass
    class Result:
        """
        The result of an optimization, with additional stats and debug information

        initial_values:
            The initial guess used for this optimization

        optimized_values:
            The best Values achieved during the optimization (Values with the smallest error)

        iteration_stats:
            Per-iteration stats, if requested, like the error per iteration.  If debug stats are
            turned on, also the Values and linearization per iteration.

        early_exited:
            Did the optimization early exit?  This can happen because it converged successfully,
            of because it was unable to make progress

        best_index:
            The index into iteration_stats for the iteration that produced the smallest error.  I.e.
            `result.iteration_stats[best_index].values == optimized_values`.  This is not guaranteed
            to be the last iteration, if the optimizer tried additional steps which did not reduce
            the error
        """

        initial_values: Values
        optimized_values: Values
        iteration_stats: T.Sequence[optimization_iteration_t]
        early_exited: bool
        best_index: int

        def error(self) -> float:
            return self.iteration_stats[self.best_index].new_error

    def __init__(
        self,
        factors: T.Iterable[T.Union[Factor, NumericFactor]],
        optimized_keys: T.Sequence[str] = None,
        params: Optimizer.Params = None,
        debug_stats: bool = False,
    ):

        if optimized_keys is None:
            # This will be filled with the optimized keys of the numeric factors
            self.optimized_keys = []
        else:
            self.optimized_keys = list(optimized_keys)
            assert len(optimized_keys) == len(
                set(optimized_keys)
            ), f"Duplicates in optimized keys: {optimized_keys}"

        numeric_factors = []
        for factor in factors:
            if isinstance(factor, Factor):
                if optimized_keys is None:
                    raise ValueError(
                        "You must specify keys to optimize when passing symbolic factors."
                    )
                # We compute the linearization in the same order as `optimized_keys`
                # so that e.g. columns of the generated jacobians are in the same order
                factor_opt_keys = [opt_key for opt_key in optimized_keys if opt_key in factor.keys]
                numeric_factors.append(factor.to_numeric_factor(factor_opt_keys))
            else:
                # Add unique keys to optimized keys
                self.optimized_keys.extend(
                    opt_key
                    for opt_key in factor.optimized_keys
                    if opt_key not in self.optimized_keys
                )
                numeric_factors.append(factor)

        # Set default params if none given
        if params is None:
            self.params = Optimizer.Params()
        else:
            self.params = params

        self.debug_stats = debug_stats

        self._initialized = False

        # Create a mapping from python identifier string keys to fixed-size C++ Key objects
        # Initialize the keys map with the optimized keys, which are needed to construct the factors
        # This works because the factors maintain a reference to this, so everything is fine as long
        # as the unoptimized keys are also in here before we attempt to linearize any of the factors
        self._cc_keys_map = {key: cc_sym.Key("x", i) for i, key in enumerate(self.optimized_keys)}

        # This stores the list of keys in the python Values, which are necessary for reconstructing
        # a Python Values from C++, in particular for methods that don't otherwise have a Python
        # Values available.  It's filled out in `_initialize`.  The order is important here, which
        # why we can't just pull the keys out of `_cc_keys_map`, which is constructed out-of-order.
        self.values_keys_ordered: T.Optional[T.List[str]] = None

        # Construct the C++ optimizer
        self._cc_optimizer = cc_sym.Optimizer(
            optimizer_params_t(**dataclasses.asdict(self.params)),
            [factor.cc_factor(self._cc_keys_map) for factor in numeric_factors],
            debug_stats=self.debug_stats,
        )

    def _initialize(self, values: Values) -> None:
        # Add unoptimized keys into the keys map
        for i, key in enumerate(values.keys_recursive()):
            if key not in self._cc_keys_map:
                # Give these a different name (`v`) so we don't have to deal with numbering
                self._cc_keys_map[key] = cc_sym.Key("v", i)

        self.values_keys_ordered = list(values.keys_recursive())

        self._initialized = True

    def _cc_values(self, values: Values) -> cc_sym.Values:
        """
        Create a cc_sym.Values from the given Python Values

        This uses the stored cc_keys_map, which will be initialized if it does not exist yet.
        """
        values = values.to_numerical()

        if not self._initialized:
            self._initialize(values)

        cc_values = cc_sym.Values()
        for key, cc_key in self._cc_keys_map.items():
            cc_values.set(cc_key, values[key])

        return cc_values

    def optimize(self, initial_guess: Values) -> Optimizer.Result:
        """
        Optimize from the given initial guess, and return the optimized Values and stats

        Args:
            initial_guess: A Values containing the initial guess, should contain at least all the
                           keys required by the `factors` passed to the constructor

        Returns:
            The optimization results, with additional stats and debug information.  See the
            `Optimizer.Result` documentation for more information
        """
        cc_values = self._cc_values(initial_guess)

        try:
            stats = self._cc_optimizer.optimize(cc_values)
        except ZeroDivisionError as ex:
            raise ZeroDivisionError("ERROR: Division by zero - check your use of epsilon!") from ex

        optimized_values = Values(
            **{
                key: cc_values.at(self._cc_keys_map[key])
                for key in initial_guess.dataclasses_to_values().keys_recursive()
            }
        )

        return Optimizer.Result(
            initial_values=initial_guess,
            optimized_values=optimized_values,
            iteration_stats=stats.iterations,
            best_index=stats.best_index,
            early_exited=stats.early_exited,
        )

    def linearize(self, values: Values) -> cc_sym.Linearization:
        """
        Compute and return the linearization at the given Values
        """
        return self._cc_optimizer.linearize(self._cc_values(values))

    def load_iteration_values(self, values_msg: values_t) -> Values:
        """
        Load a values_t message into a Python Values by first creating a C++ Values, then
        converting back to the python key names.
        """
        cc_values = cc_sym.Values(values_msg)

        assert self.values_keys_ordered is not None

        py_values = Values(
            **{key: cc_values.at(self._cc_keys_map[key]) for key in self.values_keys_ordered}
        )

        return py_values

    def linearization_index(self) -> T.Dict[str, index_entry_t]:
        """
        Get the index mapping keys to their positions in the linearized state vector.  Useful for
        extracting blocks from the problem jacobian, hessian, or RHS

        Returns: The index for the Optimizer's problem linearization
        """
        index = self._cc_optimizer.linearization_index()
        return {key: index[self._cc_keys_map[key]] for key in self.optimized_keys}

    def linearization_index_entry(self, key: str) -> index_entry_t:
        """
        Get the index entry for a given key in the linearized state vector.  Useful for extracting
        blocks from the problem jacobian, hessian, or RHS

        Args:
            key: The string key for a variable in the Python Values

        Returns: The index entry for the variable in the Optimizer's problem linearization
        """
        return self._cc_optimizer.linearization_index_entry(self._cc_keys_map[key])
