# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import itertools
import re

import symforce.symbolic as sf
from symforce import logger
from symforce import ops
from symforce import typing as T
from symforce.codegen import codegen_config
from symforce.codegen.backends.cpp.cpp_config import CppConfig
from symforce.codegen.backends.python.python_config import PythonConfig
from symforce.opt.factor import Factor
from symforce.opt.numeric_factor import NumericFactor
from symforce.opt.sub_problem import SubProblem
from symforce.values import Values


class OptimizationProblem:
    """
    An optimization problem.

    Defined by a collection of `SubProblem`s, each of which defines a set of inputs (variables in
    the Values) and a set of residuals.  SubProblems are generally expected to expose inputs that
    are used by other subproblems; these dependencies should be handled by the user while
    constructing the `residual_blocks` argument. Typical workflow is to construct a set of
    SubProblems (which should also construct each SubProblem Inputs), build the `residual_blocks`
    Values by calling `build_residuals` on each subproblem with the appropriate arguments, and then
    pass the subproblems and `residual_blocks` to the `Problem` constructor.

    Args:
        subproblems: Mapping from subproblem names to subproblems
        residual_blocks: Values where each leaf is a ResidualBlock, containing all the residuals for
                         the problem.  Typically created by calling `build_residuals` on each
                         subproblem.
        shared_inputs: If provided, an additional shared_inputs block to be added to the Values
    """

    subproblems: T.Mapping[str, SubProblem]
    inputs: Values
    residual_blocks: Values
    residuals: Values
    extra_values: Values

    def __init__(
        self,
        subproblems: T.Mapping[str, SubProblem],
        residual_blocks: Values,
        shared_inputs: T.Optional[T.Dataclass] = None,
    ):
        self.subproblems = subproblems
        self.inputs = build_inputs(self.subproblems.values(), shared_inputs)
        self.residual_blocks = residual_blocks
        self.residuals, self.extra_values = self.split_residual_blocks(residual_blocks)

    @staticmethod
    def split_residual_blocks(residual_blocks: Values) -> T.Tuple[Values, Values]:
        """
        Split residual_blocks into residuals and extra_values
        """
        residuals = Values()
        extra_values = Values()

        for key, residual_block in residual_blocks.items_recursive():
            residuals[key] = residual_block.residual
            extra_values[key] = residual_block.extra_values

        return residuals, extra_values

    def keys(self) -> T.List[str]:
        """
        Compute the set of all keys specified by the subproblems
        """
        return self.inputs.dataclasses_to_values().keys_recursive()

    def optimized_keys(self) -> T.List[str]:
        """
        Compute the set of optimized keys, as specified by the subproblems
        """
        inputs = self.inputs.dataclasses_to_values()

        optimized_values = itertools.chain.from_iterable(
            subproblem.optimized_values() for subproblem in self.subproblems.values()
        )

        content_addressible_inputs = {
            tuple(ops.StorageOps.to_storage(value)): key for key, value in inputs.items_recursive()
        }

        optimized_keys = []
        for value in optimized_values:
            optimized_key = content_addressible_inputs[tuple(ops.StorageOps.to_storage(value))]
            if value != inputs[optimized_key]:
                raise TypeError(
                    f"Variable returned by `optimized_values()` ({value}) in "
                    + "subproblem does not match variable in `Inputs` of subproblem "
                    + f"({inputs[optimized_key]}) for key {optimized_key}."
                )
            optimized_keys.append(optimized_key)

        return optimized_keys

    def generate(
        self,
        output_dir: T.Openable,
        namespace: str,
        name: str,
        sparse_linearization: bool = False,
        config: CppConfig = None,
    ) -> None:
        """
        Generate everything needed to optimize `problem` in C++. This currently assumes there is
        only one factor generated for the optimization problem.

        Args:
            output_dir: Directory in which to output the generated files.
            namespace: Namespace used in each generated file.
            name: Name of the generated factor.
            sparse_linearization: Whether the generated factors should use sparse jacobians/hessians
            config: C++ code configuration used with the linearization functions generated for each
                factor.
        """
        if config is None:
            config = CppConfig()

        # Generate the C++ code for the residual linearization function
        factors = self.make_symbolic_factors(name=name, config=config)
        for factor in factors:
            output_data = factor.generate(
                optimized_keys=self.optimized_keys(),
                output_dir=output_dir,
                namespace=namespace,
                sparse_linearization=sparse_linearization,
            )
            logger.info(
                "Generated function `{}` in directory `{}`".format(
                    output_data["name"], output_data["function_dir"]
                )
            )

    def make_symbolic_factors(
        self,
        name: str,
        config: T.Optional[codegen_config.CodegenConfig] = None,
    ) -> T.List[Factor]:
        """
        Return a list of symbolic factors for this problem for analysis purposes. If the factors
        are to be passed to an optimizer, use `make_numeric_factors` instead.

        Args:
            name: Name of factors. Note that the generated linearization functions will have
                "_factor" appended to the function name (see
                Codegen._pick_name_for_function_with_derivatives for details).
            config: Language the factors will be generated in when `.generate()` is called.
        """
        if config is None:
            config = PythonConfig()

        inputs = self.inputs.dataclasses_to_values()

        leading_trailing_dots_and_brackets_regex = re.compile(r"^[\.\[\]]+|[\.\[\]]+$")
        dots_and_brackets_regex = re.compile(r"[\.\[\]]+")

        def dots_and_brackets_to_underscores(s: str) -> str:
            """
            Converts all "." and "[]" in the given string to underscores such that the resulting
            string is a valid/readable variable name.
            """
            return re.sub(
                dots_and_brackets_regex,
                "_",
                re.sub(leading_trailing_dots_and_brackets_regex, "", s),
            )

        def compute_jacobians(keys: T.Iterable[str]) -> sf.Matrix:
            """
            Functor that computes the jacobians of the residual with respect to a set of keys

            The set of keys is not known when make_symbolic_factors is called, because we may want
            to create a NumericFactor which computes derivatives with respect to different sets of
            optimized variables.
            """
            jacobians = [
                residual_block.compute_jacobians(
                    [inputs[key] for key in keys], residual_name=residual_key, key_names=keys
                )
                for residual_key, residual_block in self.residual_blocks.items_recursive()
            ]
            return sf.Matrix.block_matrix(jacobians)

        return [
            Factor.from_inputs_and_residual(
                keys=self.keys(),
                inputs=Values(
                    **{
                        dots_and_brackets_to_underscores(key): value
                        for key, value in inputs.items_recursive()
                    }
                ),
                residual=sf.M(self.residuals.to_storage()),
                config=config,
                custom_jacobian_func=compute_jacobians,
                name=name,
            )
        ]

    def make_numeric_factors(
        self, name: str, optimized_keys: T.Sequence[str] = None
    ) -> T.List[NumericFactor]:
        """
        Returns a list of `NumericFactor`s for this problem, for example to pass to `Optimizer`.

        Args:
            name: Name of factors. Note that the generated linearization functions will have
                "_factor" appended to the function name (see
                Codegen._pick_name_for_function_with_derivatives for details).
            optimized_keys: List of keys to optimize with respect to. Defaults to the optimized keys
                specified by the subproblems of this optimization problem.
        """
        if optimized_keys is None:
            optimized_keys = self.optimized_keys()
        numeric_factors = []
        # we temp generate the factors, so skip formating to save time
        config = PythonConfig(autoformat=False)
        for factor in self.make_symbolic_factors(name, config=config):
            factor_optimized_keys = [
                opt_key for opt_key in optimized_keys if opt_key in factor.keys
            ]
            numeric_factors.append(factor.to_numeric_factor(factor_optimized_keys))
        return numeric_factors


def build_inputs(
    subproblems: T.Iterable[SubProblem], shared_inputs: T.Optional[T.Element] = None
) -> Values:
    """
    Build the inputs Values for a set of subproblems.  The resulting values is structured as:

        Values(
            subproblem1.name=subproblem1.inputs,
            ...,
            subproblemN.name=subproblemN.inputs,
            shared_inputs=shared_inputs,
        )

    Args:
        subproblems: Iterable of SubProblems
        shared_inputs: Optional additional shared inputs

    Returns:
        inputs: the combined Values
    """
    inputs = Values()

    if shared_inputs is not None:
        inputs["shared_inputs"] = shared_inputs

    # Build inputs
    for subproblem in subproblems:
        if subproblem.inputs:
            inputs[subproblem.name] = subproblem.inputs

    return inputs
