# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from dataclasses import dataclass
from dataclasses import field

import symforce.symbolic as sf
from symforce import jacobian_helpers
from symforce import typing as T


@dataclass
class ResidualBlock:
    """
    A single residual vector, with associated extra values.  The extra values are not used in the
    optimization, but are intended to be additional outputs used for debugging or other purposes.
    """

    residual: sf.Matrix
    extra_values: T.Optional[T.Dataclass] = None

    def compute_jacobians(
        self,
        inputs: T.Sequence[T.Element],
        residual_name: str = None,
        key_names: T.Sequence[str] = None,
    ) -> T.Sequence[sf.Matrix]:
        """
        Compute the jacobians of this residual block with respect to a sequence of inputs

        Args:
            inputs: Sequence of inputs to differentiate with respect to
            residual_name: Optional human-readable name of the residual to be used for debug
                           messages
            key_names: Optional sequence of human-readable names for the inputs to be used for debug
                       messages

        Returns:
            Sequence of jacobians of the residual with respect to each entry in inputs
        """
        return jacobian_helpers.tangent_jacobians(self.residual, inputs)


@dataclass
class ResidualBlockWithCustomJacobian(ResidualBlock):
    """
    A residual block, with a custom jacobian for the residual

    This should generally only be used if you want to override the jacobian computed by SymForce,
    e.g. to stop derivatives with respect to certain variables or directions, or because the
    jacobian can be analytically simplified in a way that SymForce won't do automatically.  The
    custom_jacobians field should then be filled out with a mapping from all inputs to the residual
    which may be differentiated with respect to, to the desired jacobian of the residual with
    respect to each of those inputs.
    """

    custom_jacobians: T.Dict[T.Element, sf.Matrix] = field(default_factory=dict)

    def compute_jacobians(
        self,
        inputs: T.Sequence[T.Element],
        residual_name: str = None,
        key_names: T.Sequence[str] = None,
    ) -> T.Sequence[sf.Matrix]:
        """
        Compute the jacobians of this residual block with respect to a sequence of inputs

        Args:
            inputs: Sequence of inputs to differentiate with respect to
            residual_name: Optional human-readable name of the residual to be used for debug
                           messages
            key_names: Optional sequence of human-readable names for the inputs to be used for debug
                       messages

        Returns:
            Sequence of jacobians of the residual with respect to each entry in inputs
        """
        residual_jacobians = []
        for i, input_element in enumerate(inputs):
            if input_element in self.custom_jacobians:
                # The user provided a derivative with respect to this input
                residual_jacobians.append(self.custom_jacobians[input_element])
            else:
                # The user did not provide a derivative with respect to this input.  So,
                # compute it.  If it's nonzero, raise an error, since the user probably
                # wants to provide custom jacobians for all the variables if they
                # provided one
                residual_input_jacobian = self.residual.jacobian(input_element)
                if (
                    residual_input_jacobian
                    != sf.matrix_type_from_shape(residual_input_jacobian.shape).zero()
                ):
                    residual_name = residual_name or str(self)

                    if key_names is not None:
                        key_name = key_names[i]
                    else:
                        key_name = str(input_element)

                    raise ValueError(
                        f"The residual `{residual_name}` has a nonzero jacobian with respect to "
                        f"input `{key_name}`.  Custom jacobians were provided for this residual, "
                        "but not for this input variable.  If you wish to use the automatically "
                        "computed jacobian for this input, please compute it using "
                        "`jacobian_helpers.tangent_jacobians(residual, [input])[0]` and add it to "
                        "the custom_jacobians dictionary"
                    )
                residual_jacobians.append(residual_input_jacobian)
        return residual_jacobians
