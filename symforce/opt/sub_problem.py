# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from abc import ABC
from abc import abstractmethod

from symforce import python_util
from symforce import typing as T
from symforce.ops.impl.dataclass_storage_ops import DataclassStorageOps
from symforce.values import Values


class SubProblem(ABC):
    """
    A logical grouping of a set of variables and objective terms that use those variables

    Typical usage is to subclass SubProblem, and define an `Inputs` dataclass on your subclass for
    any variables provided by the subproblem.  Then define `build_residuals`, which should return a
    Values where each leaf is a ResidualBlock, representing the residuals for your subproblem. For
    example:

        class MySubProblem(SubProblem):
            @dataclass
            class Inputs:
                x: sf.Scalar
                pose: sf.Pose3
                objective_params: MyObjective.Params

            # Optional, but helpful for type checking
            inputs: MySubProblem.Inputs

            def build_residuals(self) -> Values:
                residual_blocks = Values()
                residual_blocks["my_objective"] = MyObjective.residual(
                    self.inputs.x, self.inputs.pose, self.inputs.objective_params
                )
                return residual_blocks


    SubProblems can also depend on variables or expressions from other subproblems; the recommended
    way to do this is to add arguments to `build_residuals` for any expressions your subproblem
    needs from other subproblems.

    Both Inputs and build_residuals must be defined, but can be empty - a SubProblem can be just a
    set of variables with no objectives (for example, variables that are used in other subproblems).
    It can also be a set of objectives with no variables, i.e. with all of its inputs coming from
    other subproblems.

    Args:
        name: (optional) The name of the subproblem, derived from the class name by default
    """

    Inputs: T.Type[T.Dataclass]

    name: str
    inputs: T.Dataclass

    def __init__(self, name: str = None):
        self.name = name or self._default_name()
        assert self.name, "SubProblem name cannot be empty"
        self.build_inputs()

    def build_inputs(self) -> None:
        """
        Build the inputs block of the subproblem, and store in self.inputs.

        The default implementation works for fixed-size Dataclasses; for dynamic-size dataclasses,
        or to customize this, override this function.
        """
        self.inputs = DataclassStorageOps.symbolic(self.Inputs, name=self.name)

    @T.any_args
    @abstractmethod
    def build_residuals(self, *args: T.Any) -> Values:
        """
        Build the residual blocks for the subproblem, and return as a Values.

        Each SubProblem subclass should define this.  Typically, the SubProblem implementation of
        this function will take additional arguments, for expressions coming from other SubProblem
        dependencies or other hyperparameters.

        Returns:
            residual_blocks: A Values of any structure, but where each leaf is a ResidualBlock
        """
        pass

    @abstractmethod
    def optimized_values(self) -> T.List[T.Any]:
        """
        Return the list of optimized values for this subproblem.  Each entry should be a leaf-level
        object in the subproblem Inputs
        """
        pass

    @classmethod
    def _default_name(cls) -> str:
        """
        Pick the default name for a SubProblem class by using the class name, minus the SubProblem
        suffix if it exists.

        Returns:
            name: The subproblem name
        """
        return python_util.camelcase_to_snakecase(
            python_util.str_removesuffix(cls.__name__, "SubProblem")
        )
