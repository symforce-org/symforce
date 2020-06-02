import collections
import numpy as np

from symforce import types as T
from symforce.codegen import CodegenMode
from symforce.values import Values


class EvaluatorCodegenSpec(object):
    """
    Input specification created by the main codegen function and passed to many
    underlying routines. It contains a cache of shared information.
    """

    def __init__(
        self,
        name,  # type: str
        inputs,  # type: Values
        outputs,  # type: Values
        mode,  # type: CodegenMode
        output_dir,  # type: str
        scalar_type,  # type: T.Sequence[str]
    ):
        # type: (...) -> None
        """
        Create spec from input/output values and codegen metadata.
        """

        # Store args
        self.name = name
        self.inputs = inputs
        self.mode = mode
        self.output_dir = output_dir
        self.scalar_type = scalar_type

        # Compute values indices once to share
        self.input_values_recursive, inputs_index = inputs.flatten()
        self.output_values_recursive, outputs_index = outputs.flatten()

        # Compute type information for common use
        self.types = (
            collections.OrderedDict()
        )  # type: collections.OrderedDict[str, T.Dict[str, T.Any]]
        self.compute_typeinfo("input_t", inputs_index, self.types)
        self.compute_typeinfo("output_t", outputs_index, self.types)

    @classmethod
    def compute_typeinfo(
        cls,
        typename,  # type: str
        index,  # type: T.Dict
        typeinfo,  # type: T.Dict[str, T.Dict[str, T.Any]]
    ):
        # type: (...) -> None
        """
        Recursively compute type information from the values index and top level name. This
        data contains common information used by templates for code generation.

        Args:
            typename: Name of the type representing the current index (example: 'input_t')
            index: :func:`Values.index()`
            typeinfo: Fills in type information here
        """
        data = {}  # type: T.Dict[str, T.Any]
        data["typename"] = typename
        data["index"] = index
        data["keys_recursive"] = Values.keys_recursive_from_index(index)
        data["storage_dims"] = {key: np.prod(info[2]) for key, info in index.items()}

        # Process child types
        data["subtypes"] = {}
        for subkey, (_, datatype, _, item_index) in index.items():
            if not datatype == "Values":
                continue

            data["subtypes"][subkey] = "{}_{}_t".format(typename[:-2], subkey)

            cls.compute_typeinfo(
                typename=data["subtypes"][subkey], index=item_index, typeinfo=typeinfo
            )

        typeinfo[typename] = data
