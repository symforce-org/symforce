# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import dataclasses

from symforce import ops
from symforce import typing as T
from symforce import typing_util
from symforce.opt.sub_problem import SubProblem


class TimestepSubProblem(SubProblem):
    """
    A SubProblem intended for use when the Inputs block contains sequences tied to timesteps.

    Provides a `self.timesteps` variable for the number of timesteps, and a `build_inputs` function
    which works for Inputs blocks containing sequences as long as the number of timesteps.

    Args:
        timesteps: The number of timesteps
        name: (optional) The name of the subproblem, derived from the class name by default
    """

    timesteps: int

    def __init__(self, timesteps: int, name: str = None) -> None:
        self.timesteps = timesteps
        super().__init__(name=name)

    def build_inputs(self) -> None:
        """
        Build the inputs block of the subproblem, and store in self.inputs.

        Each field in the subproblem Inputs that's meant to be a sequence of length `self.timesteps`
        should be marked with `"timestepped": True` in the field metadata. Other sequences of known
        length should be marked with the `"length": <sequence length>` in the field metadata, where
        `<sequence length>` is the length of the sequence. For example:

            @dataclass
            class Inputs:
                my_timestepped_field: T.Sequence[sf.Scalar] = field(metadata={"timestepped": True})
                my_sequence_field: T.Sequence[sf.Scalar] = field(metadata={"length": 3})

        Any remaining fields of unknown size will cause an exception.
        """
        constructed_fields = {}

        type_hints_map = T.get_type_hints(self.Inputs)
        for field in dataclasses.fields(self.Inputs):
            field_type = type_hints_map[field.name]

            if field.metadata.get("timestepped", False):
                field_type = T.get_args(field_type)[0]
                constructed_fields[field.name] = [
                    ops.StorageOps.symbolic(field_type, f"{self.name}.{field.name}[{i}]")
                    for i in range(self.timesteps)
                ]
            elif field.metadata.get("length", False):
                sequence_instance = typing_util.get_sequence_from_dataclass_sequence_field(
                    field, field_type
                )
                constructed_fields[field.name] = ops.StorageOps.symbolic(
                    sequence_instance, f"{self.name}.{field.name}"
                )
            else:
                try:
                    constructed_fields[field.name] = ops.StorageOps.symbolic(
                        field_type, f"{self.name}.{field.name}"
                    )
                except NotImplementedError as ex:
                    raise TypeError(
                        f"Could not create instance of type {field_type} for field "
                        f"{self.name}.{field.name}; if this is a sequence, please either annotate "
                        "with timestepped=True, or override build_inputs"
                    ) from ex

        self.inputs = self.Inputs(**constructed_fields)
