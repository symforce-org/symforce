# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from lcmtypes.sym._index_entry_t import index_entry_t
from lcmtypes.sym._levenberg_marquardt_solver_failure_reason_t import (
    levenberg_marquardt_solver_failure_reason_t,
)
from lcmtypes.sym._optimization_iteration_t import optimization_iteration_t
from lcmtypes.sym._optimization_status_t import optimization_status_t
from lcmtypes.sym._optimizer_params_t import optimizer_params_t
from lcmtypes.sym._sparse_matrix_structure_t import sparse_matrix_structure_t
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
    call :meth:`optimize` repeatedly with a :class:`Values <symforce.values.values.Values>`.

    Example creation with a single :class:`Factor <.factor.Factor>`::

        factor = Factor(
            [my_key_0, my_key_1, my_key_2], my_residual_function
        )
        optimizer = Optimizer(
            factors=[factor],
            optimized_keys=[my_key_0, my_key_1],
        )

    And usage::

        initial_guess = Values(...)
        result = optimizer.optimize(initial_guess)
        print(result.optimized_values)

    Example creation with an :class:`.optimization_problem.OptimizationProblem` using
    :meth:`make_numeric_factors() <.optimization_problem.OptimizationProblem.make_numeric_factors>`.
    The linearization functions are generated in ``make_numeric_factors()`` and are linearized with
    respect to
    :meth:`problem.optimized_keys() <.optimization_problem.OptimizationProblem.optimized_keys>`::

        problem = OptimizationProblem(subproblems=[...], residual_blocks=...)
        factors = problem.make_numeric_factors("my_problem")
        optimizer = Optimizer(factors)

    Example creation with an :class:`.optimization_problem.OptimizationProblem` using
    :meth:`make_symbolic_factors() <.optimization_problem.OptimizationProblem.make_symbolic_factors>`.
    The symbolic factors are converted into numeric factors when the optimizer is created, and are
    linearized with respect to the "optimized keys" passed to the optimizer. The linearization
    functions are generated when converting to numeric factors when the optimizer is created::

        problem = OptimizationProblem(subproblems=[...], residual_blocks=...)
        factors = problem.make_symbolic_factors("my_problem")
        optimizer = Optimizer(factors, problem.optimized_keys())

    Wraps the C++ ``sym::Optimizer`` class in ``opt/optimizer.h``, so the API is mostly the same and
    optimization results will be identical.

    Args:
        factors: A sequence of either Factor or NumericFactor objects representing the
            residuals in the problem. If (symbolic) Factors are passed, they are convered to
            NumericFactors by generating linearization functions of the residual with respect to the
            keys in ``optimized_keys``.
        optimized_keys: A set of the keys to be optimized. Only required if symbolic factors are
            passed to the optimizer.
        params: Params for the optimizer
        debug_stats: Whether the optimizer should record debugging stats such as the optimized
            values, residual, jacobian, etc. computed at each iteration of the optimization.
        include_jacobians: Whether the optimizer should compute jacobians (required for linear
            error)
    """

    @dataclass
    class Params:
        """
        Parameters for the Python Optimizer

        Mirrors the ``optimizer_params_t`` LCM type, see documentation there for information on each
        parameter.

        Note: For the Python optimizer, verbose defaults to True
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

    Status = optimization_status_t
    FailureReason = levenberg_marquardt_solver_failure_reason_t

    @dataclass
    class Result:
        """
        The result of an optimization, with additional stats and debug information

        Attributes:
            initial_values:
                The initial guess used for this optimization

            optimized_values:
                The best Values achieved during the optimization (Values with the smallest error)

            iterations:
                Per-iteration stats, if requested, like the error per iteration.  If debug stats are
                turned on, also the Values and linearization per iteration.

            best_index:
                The index into iterations for the iteration that produced the smallest error.  I.e.
                ``result.iterations[best_index].values == optimized_values``.  This is not
                guaranteed to be the last iteration, if the optimizer tried additional steps which
                did not reduce the error

            status:
                What was the result of the optimization? (did it converge, fail, etc.)

            failure_reason:
                If status == FAILED, why?

            best_linearization:
                The linearization at best_index (at optimized_values), filled out if
                populate_best_linearization=True

            jacobian_sparsity:
                The sparsity pattern of the jacobian, filled out if debug_stats=True

            linear_solver_ordering:
                The ordering used for the linear solver, filled out if debug_stats=True

            cholesky_factor_sparsity:
                The sparsity pattern of the cholesky factor L, filled out if debug_stats=True
        """

        initial_values: Values
        optimized_values: Values

        # Private field holding the original stats - we expose fields of this through properties,
        # since some of the conversions out of this are expensive
        _stats: cc_sym.OptimizationStats

        @cached_property
        def iterations(self) -> T.List[optimization_iteration_t]:
            return self._stats.iterations

        @cached_property
        def best_index(self) -> int:
            return self._stats.best_index

        @cached_property
        def status(self) -> optimization_status_t:
            return self._stats.status

        @cached_property
        def failure_reason(self) -> levenberg_marquardt_solver_failure_reason_t:
            return Optimizer.FailureReason(self._stats.failure_reason)

        @cached_property
        def best_linearization(self) -> T.Optional[cc_sym.Linearization]:
            return self._stats.best_linearization

        @cached_property
        def jacobian_sparsity(self) -> sparse_matrix_structure_t:
            return self._stats.jacobian_sparsity

        @cached_property
        def linear_solver_ordering(self) -> np.ndarray:
            return self._stats.linear_solver_ordering

        @cached_property
        def cholesky_factor_sparsity(self) -> sparse_matrix_structure_t:
            return self._stats.cholesky_factor_sparsity

        @cached_property
        def iteration_stats(self) -> T.Sequence[optimization_iteration_t]:
            warnings.warn("iteration_stats is deprecated, use iterations", FutureWarning)
            return self.iterations

        def error(self) -> float:
            """
            The lowest error achieved by the optimization (the error at optimized_values)
            """
            return self.iterations[self.best_index].new_error

    def __init__(
        self,
        factors: T.Iterable[T.Union[Factor, NumericFactor]],
        optimized_keys: T.Sequence[str] = None,
        params: Optimizer.Params = None,
        debug_stats: bool = False,
        include_jacobians: bool = False,
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
                if not factor_opt_keys:
                    raise ValueError(
                        f"Factor {factor.name} has no arguments (keys: {factor.keys}) in "
                        + f"optimized_keys ({optimized_keys})."
                    )
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
        # create the mapping from cc_keys back into python keys
        self._py_keys_from_cc_keys_map = {v: k for k, v in self._cc_keys_map.items()}

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
            include_jacobians=include_jacobians,
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

    def compute_all_covariances(self, optimized_value: Values) -> T.Dict[str, np.ndarray]:
        """
        Compute the covariance matrix (J^T@J)^-1 for all optimized keys about a given linearization point

        Args:
            optimized_value: A value containing the linearization point to compute the covariance matrix about

        Returns:
            A dict of {optimized_key: numerical covariance matrix}
        """
        cc_covariance_dict = self._cc_optimizer.compute_all_covariances(
            linearization=self.linearize(optimized_value)
        )
        return {self._py_keys_from_cc_keys_map[k]: v for k, v in cc_covariance_dict.items()}

    def optimize(self, initial_guess: Values, **kwargs: T.Any) -> Optimizer.Result:
        """
        Optimize from the given initial guess, and return the optimized Values and stats

        Args:
            initial_guess: A Values containing the initial guess, should contain at least all the
                keys required by the ``factors`` passed to the constructor
            num_iterations: If < 0 (the default), uses the number of iterations specified by the
                params at construction
            populate_best_linearization: If true, the linearization at the best values will be
                filled out in the stats

        Returns:
            The optimization results, with additional stats and debug information.  See the
            :class:`Optimizer.Result` documentation for more information
        """
        cc_values = self._cc_values(initial_guess)

        try:
            stats = self._cc_optimizer.optimize(cc_values, **kwargs)
        except ZeroDivisionError as ex:
            raise ZeroDivisionError("ERROR: Division by zero - check your use of epsilon!") from ex

        optimized_values = Values(
            **{
                key: cc_values.at(self._cc_keys_map[key])
                for key in initial_guess.dataclasses_to_values().keys_recursive()
            }
        )

        return Optimizer.Result(
            initial_values=initial_guess, optimized_values=optimized_values, _stats=stats
        )

    def linearize(self, values: Values) -> cc_sym.Linearization:
        """
        Compute and return the linearization at the given Values
        """
        return self._cc_optimizer.linearize(self._cc_values(values))

    def load_iteration_values(self, values_msg: values_t) -> Values:
        """
        Load a ``values_t`` message into a Python :class:`Values <symforce.values.values.Values>`
        by first creating a C++ Values, then converting back to the python key names.
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
