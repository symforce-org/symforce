# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import itertools
from pathlib import Path
import re

from symforce import geo
from symforce import ops
from symforce.opt.factor import Factor
from symforce.opt.sub_problem import SubProblem
from symforce import typing as T
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

        return [
            content_addressible_inputs[tuple(ops.StorageOps.to_storage(value))]
            for value in optimized_values
        ]

    def generate(self, output_dir: Path) -> None:
        """
        Generate everything needed to optimize `problem` in C++
        """
        raise NotImplementedError()

    def make_factors(self, name: str) -> T.List[Factor]:
        """
        Return a list of `Factor`s for this problem, for example to pass to `Optimizer`
        """
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

        def compute_jacobians(keys: T.Iterable[str]) -> geo.Matrix:
            """
            Functor that computes the jacobians of the residual with respect to a set of keys

            The set of keys is not known when make_factors is called, because we may want to create
            a Factor and then compute derivatives with respect to different sets of optimized
            variables.
            """
            jacobians = [
                residual_block.compute_jacobians(
                    [inputs[key] for key in keys], residual_name=residual_key, key_names=keys
                )
                for residual_key, residual_block in self.residual_blocks.items_recursive()
            ]
            return geo.Matrix.block_matrix(jacobians)

        return [
            Factor(
                keys=inputs.keys_recursive(),
                name=f"{name}_factor",
                inputs=Values(
                    **{
                        dots_and_brackets_to_underscores(key): value
                        for key, value in inputs.items_recursive()
                    }
                ),
                outputs=Values(residual=geo.M(self.residuals.to_storage())),
                custom_jacobian_func=compute_jacobians,
            )
        ]


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
