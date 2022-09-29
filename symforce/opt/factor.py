# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import logging
import uuid
from pathlib import Path

import graphviz

import symforce.symbolic as sf
from symforce import logger
from symforce import python_util
from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen import codegen_config
from symforce.codegen.backends.python.python_config import PythonConfig
from symforce.codegen.similarity_index import SimilarityIndex
from symforce.opt._internal.generated_residual_cache import GeneratedResidualCache
from symforce.opt.numeric_factor import NumericFactor
from symforce.values import Values


class Factor:
    """
    A class used to represent symbolic factors as used in a factor graph. The factor is typically
    constructed from either a function or from a symbolic expression using
    `Factor.from_inputs_and_residual()`. A linearization function can be generated from the factor
    using `generate()` which can be used in a larger optimization.

    Args:
        keys: The set of variables that are inputs to the factor. These should be in order of
            the residual function arguments.
        residual: The residual function for the factor. When passed symbolic inputs, this should
            return a symbolic expression for the residual.
        config: The language the numeric factor will be generated in. Defaults to Python, which
            does not require any compilation. Also does not autoformat by default in order to
            speed up code generation.
        custom_jacobian_func: A functor that computes the jacobian, typically unnecessary unless
            you want to override the jacobian computed by SymForce, e.g. to stop derivatives
            with respect to certain variables or directions, or because the jacobian can be
            analytically simplified in a way that SymForce won't do automatically. If not
            provided, the jacobian will be computed automatically.  If provided, this should be
            a function that takes the set of optimized keys, and returns the jacobian of the
            residual with respect to those keys.
        kwargs: Any other arguments to be passed to the codegen object used to generate the
            numeric factor. See `Codegen.function()` for details.
    """

    _generated_residual_cache = GeneratedResidualCache()

    def __init__(
        self,
        keys: T.Sequence[str],
        residual: T.Callable[..., sf.Matrix],
        config: codegen_config.CodegenConfig = None,
        custom_jacobian_func: T.Callable[[T.Iterable[str]], sf.Matrix] = None,
        **kwargs: T.Any,
    ) -> None:
        # We use `_initialize()` to set `self.codegen` because we want the default constructor of
        # `Factor` to take a residual function, but the default constructor of a codegen object
        # takes inputs and outputs Values. Thus, `from_inputs_and_residual()` does not need to
        # call `__init__()`, and can instead call `__new__()` + `_initialize()` and pass its own
        # codegen object constructed using the default codegen object constructor.
        if config is None:
            config = PythonConfig(autoformat=False)
        self._initialize(
            keys=keys,
            codegen_obj=Codegen.function(func=residual, config=config, **kwargs),
            custom_jacobian_func=custom_jacobian_func,
        )

    @classmethod
    def from_inputs_and_residual(
        cls,
        keys: T.Sequence[str],
        inputs: Values,
        residual: sf.Matrix,
        config: codegen_config.CodegenConfig = None,
        custom_jacobian_func: T.Callable[[T.Iterable[str]], sf.Matrix] = None,
        **kwargs: T.Any,
    ) -> Factor:
        """
        Constructs a Factor from a set of inputs and a residual vector that consists of
        symbolic expressions composed from the symbolic variables in the inputs.

        Args:
            keys: The set of variables that are inputs to the factor. These should be in order of
                input Values entries (computed using inputs.keys_recursive()), and are the keys used
                in the overall optimization problem.
            inputs: The inputs Values for the factor.
            residual: An expression representing the residual of the factor.
            config: The language the numeric factor will be generated in. Defaults to Python, which
                does not require any compilation. Also does not autoformat by default in order to
                speed up code generation.
            custom_jacobian_func: A functor that computes the jacobian, typically unnecessary unless
                you want to override the jacobian computed by SymForce, e.g. to stop derivatives
                with respect to certain variables or directions, or because the jacobian can be
                analytically simplified in a way that SymForce won't do automatically. If not
                provided, the jacobian will be computed automatically.  If provided, this should be
                a function that takes the set of optimized keys, and returns the jacobian of the
                residual with respect to those keys.
            kwargs: Any other arguments to be passed to the codegen object used to generate the
                numeric factor. See `Codegen.__init__()` for details.
        """
        if config is None:
            config = PythonConfig(autoformat=False)
        instance = cls.__new__(cls)
        instance._initialize(
            keys=keys,
            codegen_obj=Codegen(
                inputs=inputs, outputs=Values(residual=residual), config=config, **kwargs
            ),
            custom_jacobian_func=custom_jacobian_func,
        )
        return instance

    def _initialize(
        self,
        keys: T.Sequence[str],
        codegen_obj: Codegen,
        custom_jacobian_func: T.Callable[[T.Iterable[str]], sf.Matrix] = None,
    ) -> None:
        """
        Initializes the Factor object from a codegen object

        Args:
            keys: The set of variables that are inputs to the factor.
            codegen_obj: Codegen object used to generate the numerical factor.
            custom_jacobian_func: Custom jacobian function as described in `__init__`.
        """
        if len(codegen_obj.inputs.keys()) != len(codegen_obj.inputs.keys_recursive()):
            raise ValueError(
                "Only flat inputs are currently supported (i.e. inputs should not"
                + " contain nested Values objects or Dataclasses or Sequences."
            )

        if len(keys) != len(codegen_obj.inputs.keys_recursive()):
            raise ValueError(
                "There must be a key for each input to the residual. Expected"
                + f" {len(codegen_obj.inputs.keys_recursive())} keys but got {len(keys)} keys."
            )

        self.keys = keys
        self.codegen = codegen_obj
        self.custom_jacobian_func = custom_jacobian_func
        self.name = self.codegen.name
        self.generated_jacobians: T.Dict[T.Tuple[str, ...], sf.Matrix] = {}

    def generate(
        self,
        optimized_keys: T.Sequence[str],
        output_dir: T.Openable = None,
        namespace: str = None,
        sparse_linearization: bool = False,
    ) -> T.Dict[str, T.Any]:
        """
        Generates the code needed to construct a NumericFactor from this Factor.

        Args:
            optimized_keys: Keys which we compute the linearization of the residual with respect to.
            output_dir: Where the generated linearization function will be output.
            namespace: Namespace of the generated linearization function.
            sparse_linearization: Whether the generated linearization function should use sparse
                matricies for the jacobian and hessian approximation

        Returns:
            Dict containing locations where the code was generated (e.g. "output_dir" and
            "python_function_dir" or "cpp_function_dir") and the name of the generated function.
        """
        if namespace is None:
            namespace = f"sym_{uuid.uuid4().hex}"

        codegen_keys = list(self.codegen.inputs.keys())
        codegen_with_linearization = self.codegen.with_linearization(
            which_args=[codegen_keys[self.keys.index(key)] for key in optimized_keys],
            custom_jacobian=self.custom_jacobian_func(optimized_keys)
            if self.custom_jacobian_func is not None
            else None,
            sparse_linearization=sparse_linearization,
        )
        # Ignore false positive because we define `self.jacobian` in `_initialize()`
        # pylint: disable=attribute-defined-outside-init
        self.generated_jacobians[tuple(optimized_keys)] = codegen_with_linearization.outputs[
            "jacobian"
        ]

        output_data = codegen_with_linearization.generate_function(
            output_dir=output_dir, namespace=namespace
        )

        metadata = dataclasses.asdict(output_data)
        metadata["name"] = codegen_with_linearization.name

        return metadata

    def to_numeric_factor(
        self,
        optimized_keys: T.Sequence[str],
        output_dir: T.Openable = None,
        namespace: str = None,
        sparse_linearization: bool = False,
    ) -> NumericFactor:
        """
        Constructs a NumericFactor from this Factor, including generating a linearization
        function.

        Args:
            optimized_keys: Keys which we compute the linearization of the residual with respect to.
            output_dir: Where the generated linearization function will be output
            namespace: Namespace of the generated linearization function
            sparse_linearization: Whether the generated linearization function should use sparse
                matrices for the jacobian and hessian approximation
        """
        for opt_key in optimized_keys:
            if opt_key not in self.keys:
                raise ValueError(
                    f"Optimization key {opt_key} does not match any of the keys used as inputs"
                    + " to this factor. The optimization keys must be a subset of the input keys to"
                    + " this factor."
                )

        if not isinstance(self.codegen.config, PythonConfig):
            raise NotImplementedError(
                "We currently only support generating and then loading python factors."
            )

        # If we have already generated a factor of the same form, load the previously generated
        # factor.
        similarity_index = SimilarityIndex.from_codegen(self.codegen)
        codegen_keys = list(self.codegen.inputs.keys())
        codegen_optimized_keys = [codegen_keys[self.keys.index(key)] for key in optimized_keys]
        # NOTE(aaron): This should contain all of the information required to both generate the
        # residual, and cause any side effects of that.  Basically, this means all information
        # used to construct this Factor plus the arguments to this function
        cache_key = (
            similarity_index,
            codegen_optimized_keys,
            output_dir,
            namespace,
            sparse_linearization,
        )
        cached_residual = Factor._generated_residual_cache.get_residual(*cache_key)
        if cached_residual is not None:
            return NumericFactor(
                keys=self.keys,
                optimized_keys=optimized_keys,
                linearization_function=cached_residual,
            )

        # NOTE(aaron): We do this after checking the cache, otherwise we'd get 0 cache hits.  I
        # _think_ this is correct, since the interface of to_numeric_factor doesn't specify the
        # namespace if the user passes None, so you want to get the same namespace as when it was
        # cached
        if namespace is None:
            namespace = f"sym_{uuid.uuid4().hex}"

        # Compute the linearization of the residual and generate code
        output_data = self.generate(optimized_keys, output_dir, namespace, sparse_linearization)

        # Load the generated function
        numeric_factor = NumericFactor.from_file_python(
            keys=self.keys,
            optimized_keys=optimized_keys,
            output_dir=output_data["output_dir"],
            namespace=namespace,
            name=output_data["name"],
        )

        Factor._generated_residual_cache.cache_residual(
            *cache_key, numeric_factor.linearization_function
        )

        if output_dir is None and logger.level != logging.DEBUG:
            # We generated the function into a temp directory; delete it now that it's loaded.
            python_util.remove_if_exists(output_data["output_dir"])

        return numeric_factor


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
