# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np

from symforce import cc_sym
from symforce import typing as T
from symforce.codegen import codegen_util
from symforce.values import Values


class NumericFactor:
    """
    A class used to wrap linearization functions such that they can be used by the optimizer.

    Args:
        keys: The set of keys that are inputs to the linearization function.
        optimized_keys: A subset of `keys` representing the keys which the given linearization
                function computes the jacobian with respect to.
        linearization_function: A function that returns the residual, jacobian, hessian
            approximation, and right-hand-side used with the levenberg marquardt optimizer.
    """

    def __init__(
        self,
        keys: T.Sequence[str],
        optimized_keys: T.Sequence[str],
        linearization_function: T.Callable[
            ..., T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
    ) -> None:
        self.keys = keys
        self.optimized_keys = optimized_keys
        self.linearization_function = linearization_function

    @classmethod
    def from_file_python(
        cls,
        keys: T.Sequence[str],
        optimized_keys: T.Sequence[str],
        output_dir: T.Openable,
        namespace: str,
        name: str,
    ) -> NumericFactor:
        """
        Loads a generated python linearization function from a given file. This can be used after
        generating a linearization function from a symbolic factor as follows:

        Create a symbolic factor and generate the linearization function:
            output_dir = "my_output_dir"
            namespace = "my_namespace"
            name = "my_custom_factor"
            sym_factor = Factor(
                keys=my_keys, residual=my_func, name=name,
            )
            sym_factor.generate(my_optimized_keys, output_dir, namespace)

        Load the generated linearization function:
            num_factor = NumericFactor.from_file_python(
                my_keys, my_optimized_keys, output_dir, namespace, name
            )

        Args:
            keys: The set of keys that are inputs to the linearization function.
            optimized_keys: A subset of `keys` representing the keys which the given linearization
                function computes the jacobian with respect to.
            output_dir: The top-level output directory of the linearization function.
            namespace: The namespace of the linearization function.
            name: The name of the linearization function.
        """
        assert all(opt_key in keys for opt_key in optimized_keys)
        function_dir = Path(output_dir) / "python" / "symforce" / namespace
        linearization_function = codegen_util.load_generated_function(name, function_dir)
        return cls(
            keys=keys, optimized_keys=optimized_keys, linearization_function=linearization_function
        )

    def linearize(self, inputs: Values) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates the linearization function for the given inputs. Returns the residual, jacobian,
        hessian approximation, and right hand side used with the levenberg marquardt optimizer.

        Args:
            inputs: Values object that does not contain any symbolic members and is ordered the
                same as the arguments to the linearization function.
        """
        if inputs.keys_recursive() != self.keys:
            raise ValueError("Keys in inputs must match keys used to construct the factor.")
        return self.linearization_function(*inputs.to_numerical().values_recursive())

    def cc_factor(self, cc_key_map: T.Mapping[str, cc_sym.Key]) -> cc_sym.Factor:
        """
        Create a C++ Factor from this symbolic Factor, for use with the C++ Optimizer
        Note that while this is a C++ Factor object, the linearization function may be a compiled
        C++ function or a Python function passed into C++ through pybind, depending on
        the language the linearization function was generated in.

        Args:
            cc_key_map: Mapping from Python keys (strings, like returned by
                        `Values.keys_recursive()`) to C++ keys
        Returns:
            A C++ wrapped Factor object
        """

        def wrapped(
            values: cc_sym.Values, _: T.Any
        ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            return self.linearization_function(*[values.at(cc_key_map[key]) for key in self.keys])

        return cc_sym.Factor(wrapped, [cc_key_map[key] for key in self.optimized_keys])
