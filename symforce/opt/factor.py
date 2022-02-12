# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import enum
import graphviz
import numpy as np
from pathlib import Path
import uuid

from symforce import cc_sym
from symforce import codegen
from symforce.codegen import codegen_util
from symforce import geo
from symforce import python_util
from symforce import typing as T
from symforce.values import Values


class FactorCodegenLanguage(enum.Enum):
    PYTHON = 1
    CPP = 2


class Factor:
    """
    A symbolic Factor, representing a residual function applied to a set of keys.

    Can be used to generate numerical code for efficiently evaluating the factor in Python or C++

    Can be constructed from either a residual function, or inputs and outputs Values

    Args:
        keys: The set of variables that are inputs to the factor.  These should be in order of
              the residual function arguments or input Values entries (whichever is applicable)
        name: The name of the factor - optional for residual function Factors (will be deduced from
              the function name if not provided)
        residual: The residual function for the factor.  Either this, or both `inputs` and `outputs`
                  must be provided
        inputs: The inputs Values for the factor.  Either this and `outputs`, or just `residual`,
                must be provided
        outputs: The outputs Values for the factor.  Either this and `inputs`, or just `residual`,
                 must be provided
        codegen_language: The language in which to generate numerical code to evaluate the factor,
                          if needed.  Defaults to PYTHON, which does not require any compilation
    """

    def __init__(
        self,
        keys: T.Sequence[str],
        name: str = None,
        *,
        residual: T.Callable[..., geo.M] = None,
        inputs: Values = None,
        outputs: Values = None,
        codegen_language: FactorCodegenLanguage = FactorCodegenLanguage.PYTHON,
    ):
        if codegen_language != FactorCodegenLanguage.PYTHON:
            raise NotImplementedError

        self.name = name
        self.codegen_language = codegen_language

        if residual is not None:
            self.codegen = codegen.Codegen.function(
                residual,
                name=name,
                config=codegen.PythonConfig(
                    # NOTE(hayk): This is to speed up generation
                    autoformat=False
                ),
            )
            if self.name is None:
                self.name = self.codegen.name
        else:
            if inputs is None or outputs is None:
                raise ValueError

            if name is None:
                raise ValueError

            self.codegen = codegen.Codegen(
                inputs=inputs, outputs=outputs, name=self.name, config=codegen.PythonConfig(),
            )

        # TODO(hayk): Should we convert to a set and make sure there were no duplicates?
        self.keys = keys

        self.generated_residual = None
        self.optimized_keys: T.Optional[T.Set[str]] = None

    def __repr__(self) -> str:
        """
        Return a string representation.
        """
        return f"<Factor {self.name} ({', '.join(self.keys)})>"

    def _generate_python_function(self, optimized_keys: T.Set[str]) -> None:
        """
        Generate a numerical Python function for the factor, and store it in self.generated_residual

        If we've previously generated a numerical function, but for a different set of optimized
        keys, this will generate a new residual function for the given set of optimized keys.

        Args:
            optimized_keys: The set of keys which are optimized (keys which we need derivatives with
                            respect to).  Must be a subset of self.keys.
        """
        # This is unused, but if we don't import it things blow up.  I _think_ it's because
        # the generated residual imports it, and we import that then delete the directory
        # where it was generated...
        import sym  # pylint: disable=unused-import

        if self.generated_residual is not None and self.optimized_keys == optimized_keys:
            return

        self.optimized_keys = optimized_keys

        inputs = list(self.codegen.inputs.keys())
        codegen_with_linearization = self.codegen.with_linearization(
            which_args=[inputs[i] for i, key in enumerate(self.keys) if key in self.optimized_keys],
        )

        namespace = f"factor_{uuid.uuid4().hex}"
        codegen_data = codegen_with_linearization.generate_function(namespace=namespace)

        assert codegen_with_linearization.name is not None
        self.generated_residual = getattr(
            codegen_util.load_generated_package(
                f"{namespace}.{self.name}", codegen_data["python_function_dir"]
            ),
            codegen_with_linearization.name,
        )
        python_util.remove_if_exists(codegen_data["output_dir"])

    def cc_factor(
        self, optimized_keys: T.Set[str], cc_key_map: T.Mapping[str, cc_sym.Key]
    ) -> cc_sym.Factor:
        """
        Create a C++ Factor from this symbolic Factor, for use with the C++ Optimizer

        Note that while this is a C++ Factor object, the residual function may be a compiled C++
        function or a Python function passed into C++ through pybind, depending on
        self.codegen_language

        Args:
            optimized_keys: The set of keys which are optimized (keys which we need derivatives with
                            respect to).  Must be a subset of self.keys.
            cc_key_map: Mapping from Python keys (strings, like returned by
                        `Values.keys_recursive()`) to C++ keys

        Returns:
            A C++ wrapped Factor object
        """
        if self.codegen_language == FactorCodegenLanguage.PYTHON:
            self._generate_python_function(optimized_keys)

            # We don't use the index_entries here since cc_key_map is populated lazily by the python
            # Optimizer, so we have to bind cc_key_map in here instead of passing to the Factor
            # constructor
            def wrapped(
                values: cc_sym.Values, _: T.Any
            ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                assert self.generated_residual is not None
                return self.generated_residual(*[values.at(cc_key_map[key]) for key in self.keys])

            return cc_sym.Factor(
                wrapped, [cc_key_map[key] for key in self.keys if key in optimized_keys],
            )
        else:
            raise NotImplementedError


def visualize_factors(factors: T.Sequence[Factor], outfile: T.Openable = None) -> graphviz.Graph:
    """
    Construct a dot file of the factor graph given by the input set of factors.

    Args:
        factors: List of factors to visualize, including all mentioned keys
        outfile: Output file, if given. Format is parsed from the extension.

    Returns:
        graph object
    """
    key_to_type: T.Dict[str, type] = {}
    for factor in factors:
        for key, value in zip(factor.keys, factor.codegen.inputs.values()):
            key_to_type[key] = type(value)

    dot = graphviz.Graph(engine="dot")
    dot.attr(forcelabels="true")

    for key, value_type in key_to_type.items():
        dot.node(key, label=f"{key} : {value_type.__name__}")

    for i, factor in enumerate(factors):
        name = f"factor_{i}"
        dot.node(name, xlabel=f"{factor.name} ({i})", shape="point", height="0.15")
        for key in factor.keys:
            dot.edge(name, key)

    if outfile:
        dot.render(
            outfile=outfile, format=Path(outfile).suffix[1:], cleanup=True, overwrite_filepath=True
        )

    return dot
