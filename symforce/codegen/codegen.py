from __future__ import annotations

import enum
import os
import tempfile
import textwrap

from symforce import sympy as sm
from symforce import geo
from symforce import ops
from symforce import logger
from symforce import python_util
from symforce import types as T
from symforce.values import Values
from symforce.codegen import template_util
from symforce.codegen import codegen_util
from symforce.codegen import type_helper
from symforce.codegen import types_package_codegen

CURRENT_DIR = os.path.dirname(__file__)


class DerivativeMode(enum.Enum):
    """
    Mode for create_with_derivatives
    """

    # Compute jacobians for input arguments as separate matrices.  In this mode, the first result
    # will also be returned, not an output argument (this may be the original function result, or
    # the jacobian if only computing derivatives w.r.t one input argument and not returning the
    # original function result)
    SEPARATE_JACOBIANS = "separate_jacobians"

    # Compute jacobians for input arguments stacked into a single jacobian matrix.  In this mode,
    # the generated function will never return anything, all outputs will be output arguments.
    STACKED_JACOBIAN = "stacked_jacobian"

    # Compute a full linearization for the output with respect to the given input arguments.  This
    # includes the jacobian, hessian (computed as J^T J with only the lower triangle filled out),
    # and rhs (J^T b).  In this mode, the original function must return a vector (a geo.Matrix with
    # one column) and the generated function will never return anything, all outputs will be output
    # arguments.
    FULL_LINEARIZATION = "full_linearization"


class Codegen:
    """
    Class used for generating code from symbolic expressions or functions.

    Codegen objects can either be used to generate standalone functions, or
    as specifications in a larger code generation pipeline. Each codegen object
    defines an input/output relationship between a set of symbolic inputs and
    a set of symbolic output expressions written in terms of the inputs.
    """

    def __init__(
        self,
        inputs: Values,
        outputs: Values,
        mode: codegen_util.CodegenMode,
        name: T.Optional[str] = None,
        return_key: T.Optional[str] = None,
        sparse_matrices: T.List[str] = None,
        scalar_type: str = "double",
        docstring: str = None,
    ) -> None:
        """
        Creates the Codegen specification.

        Args:
            inputs: Values object specifying names and symbolic inputs to the function
            outputs: Values object specifying names and output expressions (written in terms
                of the symbolic inputs) of the function
            mode: Programming language in which the function is to be generated
            name: Name of the function to be generated; must be set before the function is
                generated, but need not be set here if it's going to be set by create_with_derivatives
            return_key: If specified, the output with this key is returned rather than filled
                in as a named output argument.
            sparse_matrices: Outputs with this key will be returned as sparse matrices
            scalar_type: Type used for generated scalar expressions
            docstring: The docstring to be used with the generated function
        """

        self.name = name

        # Inputs and outputs must be Values objects
        assert isinstance(inputs, Values)
        assert isinstance(outputs, Values)

        # All symbols in outputs must be present in inputs
        input_symbols = set(inputs.to_storage())
        assert all([sm.S(v).free_symbols.issubset(input_symbols) for v in outputs.to_storage()])

        # Names given by keys in inputs/outputs must be valid variable names
        # TODO(aaron): Also check recursively
        assert all([python_util.is_valid_variable_name(k) for k in inputs.keys()])
        assert all([python_util.is_valid_variable_name(k) for k in outputs.keys()])

        # Symbols in inputs must be unique
        assert len(set(inputs.to_storage())) == len(
            inputs.to_storage()
        ), "Symbols in inputs must be unique. Duplicate symbols = {}".format(
            [symbol for symbol in inputs.to_storage() if inputs.to_storage().count(symbol) > 1]
        )

        # Outputs must not have same variable names/keys as inputs
        assert all([key not in list(outputs.keys()) for key in inputs.keys()])

        self.inputs = inputs
        self.outputs = outputs

        self.mode = mode
        self.scalar_type = scalar_type

        if return_key is not None:
            assert return_key in outputs
        self.return_key = return_key

        # Mapping between sparse matrix keys and constants needed for static CSC construction
        self.sparse_mat_data: T.Dict[str, T.Dict[str, T.Any]] = {}
        if sparse_matrices is not None:
            assert all([key in outputs for key in sparse_matrices])
            assert all([isinstance(outputs[key], geo.Matrix) for key in sparse_matrices])
            for key in sparse_matrices:
                self.sparse_mat_data[key] = codegen_util.get_sparse_mat_data(outputs[key])

        self.docstring = docstring or Codegen.default_docstring(inputs=inputs, outputs=outputs)

        # TODO(nathan): Consider moving into a different function so that we can generate code separately
        (
            self.intermediate_terms,
            self.output_terms,
            self.sparse_terms,
            self.total_ops,
        ) = codegen_util.print_code(
            inputs=self.inputs,
            outputs=self.outputs,
            sparse_mat_data=self.sparse_mat_data,
            mode=self.mode,
        )

    @classmethod
    def function(
        cls,
        func: T.Callable,
        mode: codegen_util.CodegenMode,
        name: T.Optional[str] = None,
        input_types: T.Sequence[T.ElementOrType] = None,
        output_names: T.Sequence[str] = None,
        return_key: str = None,
        docstring: str = None,
    ) -> Codegen:
        """
        Creates a Codegen object from a symbolic python function.

        Args:
            func: Python function
            input_types: List of types of the inputs to the given function.  This is optional; if
                `func` has type annotations, `input_types` can be deduced from those.  Note that
                if the type annotation doesn't match what you want the arguments to be, you need
                to specify manually, for instance a function add(x: T.Any, y: T.Any) -> T.Any that
                you want to use to generate add(x: geo.Matrix33, y: geo.Matrix33) -> geo.Matrix33
            mode: Programming language in which the function is to be generated
            name: Name of the function to be generated; if not provided, will be deduced from the
                function name.  Must be provided if `func` is a lambda
            output_names: Optional if only one object is returned by the function.
                If multiple objects are returned, they must be named.
            return_key: If multiple objects are returned, the generated function will return
                the object with this name (must be in output_names)
        """
        arg_spec = codegen_util.get_function_argspec(func)

        if input_types is None:
            input_types = type_helper.deduce_input_types(func)

        if name is None:
            assert func.__name__ != "<lambda>", "Can't deduce name automatically for a lambda"
            name = func.__name__
            if mode == codegen_util.CodegenMode.CPP:
                name = python_util.snakecase_to_camelcase(name)

        # Formulate symbolic arguments to function
        assert len(arg_spec.args) == len(input_types)
        symbolic_args = []
        inputs = Values()
        for arg_name, arg_type in zip(arg_spec.args, input_types):
            inputs[arg_name] = ops.StorageOps.symbolic(arg_type, arg_name)
            symbolic_args.append(inputs[arg_name])

        # Run the symbolic arguments through the function and get the symbolic output expression(s)
        res = func(*symbolic_args)

        if isinstance(res, tuple):
            # Function returns multiple objects
            output_terms = res
            assert output_names is not None, "Must give output_names for multiple outputs"
            # If a return key is given, it must be valid (i.e. in output_names)
            if return_key is not None:
                assert return_key in output_names, "Return key not found in named outputs"
        else:
            # Function returns single object
            output_terms = (res,)
            if output_names is None:
                output_names = ["res"]
                return_key = output_names[0]
        assert len(output_terms) == len(output_names)

        # Form the output expressions as a Values object
        outputs = Values()
        for output_name, output in zip(output_names, output_terms):
            if isinstance(output, (list, tuple)):
                output = geo.Matrix(output)
            outputs[output_name] = output

        # Pull docstring out of function if not provided
        if docstring is None:
            if func.__doc__:
                docstring = func.__doc__
            else:
                docstring = Codegen.default_docstring(
                    inputs=inputs, outputs=outputs, original_function=func
                )

        return cls(
            name=name,
            inputs=inputs,
            outputs=outputs,
            mode=mode,
            return_key=return_key,
            docstring=textwrap.dedent(docstring),
        )

    @staticmethod
    def common_data() -> T.Dict[str, T.Any]:
        """
        Return common template data for code generation.
        """
        data: T.Dict[str, T.Any] = {}
        data["ops"] = ops
        data["Symbol"] = sm.Symbol
        data["Matrix"] = geo.Matrix
        data["Values"] = Values
        data["scalar_types"] = ("double", "float")
        data["camelcase_to_snakecase"] = python_util.camelcase_to_snakecase
        data["python_util"] = python_util

        def is_symbolic(T: T.Any) -> bool:
            return isinstance(T, (sm.Expr, sm.Symbol))

        data["is_symbolic"] = is_symbolic
        data["issubclass"] = issubclass
        data["is_sequence"] = lambda arg: isinstance(arg, (list, tuple))
        return data

    def generate_function(
        self,
        output_dir: str = None,
        lcm_bindings_output_dir: str = None,
        shared_types: T.Mapping[str, str] = None,
        namespace: str = "sym",
        generated_file_name: str = None,
    ) -> T.Dict[str, T.Any]:
        """
        Generates a function that computes the given outputs from the given inputs.

        Usage for generating multiple functions with a shared type:
            codegen_obj_1.generate_function(namespace="my_namespace")
            shared_types = {"my_type": "my_namespace.my_type_t"}
            codegen_obj_2.generate_function(shared_types=shared_types, namespace="my_namespace")

        In the example above, both codegen_obj_1 and codegen_obj_2 use the type "my_type". During
        the first call to "generate_function" we generate the type "my_type", and it then becomes
        a shared type for the second call to "generate_function". This signals that "my_type" does
        not need to be generated during the second call to "generate_function" as it already exists.

        Args:
            output_dir: Directory in which to output the generated function. Any generated types will
                be located in a subdirectory with name equal to the namespace argument.
            lcm_bindings_output_dir: Directory in which to output language-specific LCM bindings
            shared_types: Mapping between types defined as part of this codegen object (e.g. keys in
                self.inputs that map to Values objects) and previously generated external types.
            namespace: Namespace for the generated function and any generated types.
            generated_file_name: Stem for the filename into which the function is generated, with no file extension
        """
        assert (
            self.name is not None
        ), "Name should be set either at construction or by create_with_derivatives"

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix=f"sf_codegen_{self.name}_", dir="/tmp")
            logger.debug(f"Creating temp directory: {output_dir}")

        if lcm_bindings_output_dir is None:
            lcm_bindings_output_dir = output_dir

        if generated_file_name is None:
            generated_file_name = self.name

        # List of (template_path, output_path, data)
        templates = template_util.TemplateList()

        # Output types
        # Find each Values object in the inputs and outputs
        types_to_generate = []
        # Also keep track of non-Values types used so we can have the proper includes - things like
        # geo types and cameras
        self.types_included = set()
        for d in (self.inputs, self.outputs):
            for key, value in d.items():
                # If "value" is a list, extract an instance of a base element.
                base_value = codegen_util.get_base_instance(value)

                if isinstance(base_value, Values):
                    types_to_generate.append((key, base_value))
                else:
                    self.types_included.add(type(base_value).__name__)

        # Generate types from the Values objects in our inputs and outputs
        values_indices = {name: gen_type.index() for name, gen_type in types_to_generate}
        types_codegen_data = types_package_codegen.generate_types(
            package_name=namespace,
            file_name=python_util.camelcase_to_snakecase(generated_file_name),
            values_indices=values_indices,
            shared_types=shared_types,
            scalar_type=self.scalar_type,
            output_dir=output_dir,
            lcm_bindings_output_dir=lcm_bindings_output_dir,
            templates=templates,
        )

        # Maps typenames to generated types
        self.typenames_dict = types_codegen_data["typenames_dict"]
        # Maps typenames to namespaces
        self.namespaces_dict = types_codegen_data["namespaces_dict"]
        self.unique_namespaces = {v for v in self.namespaces_dict.values()}

        # Namespace of this function + generated types
        self.namespace = namespace

        output_data = {
            "output_dir": output_dir,
            "lcm_type_dir": types_codegen_data["lcm_type_dir"],
        }

        # Generate the function
        if self.mode == codegen_util.CodegenMode.PYTHON2:
            python_function_dir = os.path.join(output_dir, "python2.7", "symforce", namespace)
            logger.info(f'Creating python function "{self.name}" at "{python_function_dir}"')

            templates.add(
                os.path.join(template_util.PYTHON_TEMPLATE_DIR, "function", "FUNCTION.py.jinja"),
                os.path.join(python_function_dir, generated_file_name + ".py"),
                dict(self.common_data(), spec=self),
            )
            templates.add(
                os.path.join(template_util.PYTHON_TEMPLATE_DIR, "function", "__init__.py.jinja"),
                os.path.join(python_function_dir, "__init__.py"),
                dict(self.common_data(), spec=self),
            )

            output_data["python_function_dir"] = python_function_dir
        elif self.mode == codegen_util.CodegenMode.CPP:
            cpp_function_dir = os.path.join(output_dir, "cpp", "symforce", namespace)
            logger.info(f'Creating C++ function "{self.name}" at "{cpp_function_dir}"')

            templates.add(
                os.path.join(template_util.CPP_TEMPLATE_DIR, "function", "FUNCTION.h.jinja"),
                os.path.join(
                    cpp_function_dir, python_util.camelcase_to_snakecase(generated_file_name) + ".h"
                ),
                dict(self.common_data(), spec=self),
            )

            output_data["cpp_function_dir"] = cpp_function_dir
        else:
            raise NotImplementedError(f'Unknown mode: "{self.mode}"')

        templates.render()
        lcm_data = codegen_util.generate_lcm_types(
            lcm_type_dir=types_codegen_data["lcm_type_dir"],
            lcm_files=types_codegen_data["lcm_files"],
            lcm_output_dir=types_codegen_data["lcm_bindings_output_dir"],
        )
        output_data.update(lcm_data)

        output_data["generated_files"] = [v[1] for v in templates.items]

        return output_data

    @staticmethod
    def default_docstring(
        inputs: Values, outputs: Values, original_function: T.Callable = None
    ) -> str:
        """
        Create a default docstring if no other is available from the function or caller.
        """
        # If the function is an instance method, remove the type associated with the class
        input_types = [
            python_util.get_type(arg).__name__ for name, arg in inputs.items() if name != "self"
        ]
        output_types = [python_util.get_type(arg).__name__ for arg in outputs.values()]

        # TODO(nathan): This sometimes doesn't print the types in a nice way. For example,
        # scalar types are not printed as "Scalar" but instead "Symbol", "One", "Max", etc.

        if original_function is not None:
            docstring = """
            This function was autogenerated from a symbolic function. Do not modify by hand.

            Symbolic function: {}
            Arg type(s): {}
            Return type(s): {}
            """.format(
                original_function.__name__, ", ".join(input_types), ", ".join(output_types),
            )
        else:
            docstring = """
            This function was autogenerated. Do not modify by hand.

            Arg type(s): {}
            Return type(s): {}
            """.format(
                ", ".join(input_types), ", ".join(output_types),
            )
        return textwrap.dedent(docstring)

    def create_with_derivatives(
        self,
        which_args: T.Sequence[int] = None,
        include_result: bool = True,
        name: str = None,
        derivative_generation_mode: DerivativeMode = DerivativeMode.SEPARATE_JACOBIANS,
    ) -> Codegen:
        """
        Given a codegen object that takes some number of inputs and computes a single result,
        create a codegen object that additionally computes jacobians with respect to
        the given input arguments. Flexible to produce the value and all jacobians, just the
        jacobians, or any combination of one or more jacobians.

        Args:
            self: Existing codegen object that return a single value
            which_args: Indices of args for which to compute jacobians. If not given, uses all.
            include_result: Whether this codegen object computes the value in addition to jacobians
            name: Generated function name. If not given, picks a reasonable name based on the one
                                           given at construction.
            derivative_generation_mode: Whether to generate separate jacobians
                                        (SEPARATE_JACOBIANS), combine them into a single jacobian
                                        matrix (STACKED_JACOBIANS), or generate a full
                                        linearization with a hessian and rhs (FULL_LINEARIZATION).
                                        Also changes whether the result will be returned from the
                                        generated function or handled as an output argument - for
                                        SEPARATE_JACOBIANS, the result will be returned (if
                                        include_result == True), but for STACKED_JACOBIAN or
                                        FULL_LINEARIZATION it will be an output argument.
        """
        if not which_args:
            which_args = list(range(len(list(self.inputs.keys()))))

        # Get docstring
        docstring_lines = self.docstring.split("\n")[:-1]

        # Ensure the previous codegen has one output
        assert len(list(self.outputs.keys())) == 1
        result_name, result = list(self.outputs.items())[0]

        if derivative_generation_mode == DerivativeMode.FULL_LINEARIZATION:
            # Ensure the output is a vector (the residual)
            assert isinstance(result, geo.Matrix) and result.cols == 1

        # Make the new outputs
        outputs = Values()
        if include_result:
            outputs[result_name] = result
        else:
            # Remove return val line from docstring
            docstring_lines = docstring_lines[:-1]

        # Compute jacobians in the space of the storage, then chain rule on the left and right sides
        # to get jacobian wrt the tangent space of both the arg and the result
        jacobian = None
        docstring_args = []

        input_args = list(self.inputs.items())
        result_storage = geo.M(ops.StorageOps.to_storage(result))
        result_tangent_D_storage = ops.LieGroupOps.tangent_D_storage(result)
        for arg_index in which_args:
            arg_name, arg = input_args[arg_index]
            result_storage_D_arg_storage = result_storage.jacobian(ops.StorageOps.to_storage(arg))
            arg_jacobian = (
                result_tangent_D_storage
                * result_storage_D_arg_storage
                * ops.LieGroupOps.storage_D_tangent(arg)
            )

            if derivative_generation_mode in (
                DerivativeMode.STACKED_JACOBIAN,
                DerivativeMode.FULL_LINEARIZATION,
            ):
                if jacobian is None:
                    jacobian = arg_jacobian
                else:
                    jacobian = jacobian.row_join(arg_jacobian)

                docstring_args.append(f"{arg_index} ({arg_name})")
            elif derivative_generation_mode == DerivativeMode.SEPARATE_JACOBIANS:
                outputs[f"{result_name}_D_{arg_name}"] = arg_jacobian
                docstring_lines.append(f"    geo.Matrix: Jacobian for arg {arg_index} ({arg_name})")

        if derivative_generation_mode in (
            DerivativeMode.STACKED_JACOBIAN,
            DerivativeMode.FULL_LINEARIZATION,
        ):
            outputs["jacobian"] = jacobian
            docstring_lines.append(
                "    geo.Matrix: Jacobian for args {}".format(", ".join(docstring_args))
            )

        if derivative_generation_mode == DerivativeMode.FULL_LINEARIZATION:
            # Ensure that we have at least one input in the jacobian
            assert jacobian is not None

            outputs["hessian"] = jacobian.compute_AtA(lower_only=True)
            docstring_lines.append(
                "    geo.Matrix: Hessian for args {}".format(", ".join(docstring_args))
            )

            outputs["rhs"] = jacobian.T * result
            docstring_lines.append(
                "    geo.Matrix: rhs for args {}".format(", ".join(docstring_args))
            )

        # If just computing a single jacobian, return it instead of output arg
        return_key = None
        return_result = derivative_generation_mode == DerivativeMode.SEPARATE_JACOBIANS
        if len(list(outputs.keys())) == 1 or (include_result and return_result):
            return_key = list(outputs.keys())[0]

        # Cutely pick a function name if not given
        if not name:
            assert (
                self.name is not None
            ), "Codegen name must have been provided already to automatically generate a name for create_with_derivatives"

            name = self.name + "_"
            if derivative_generation_mode == DerivativeMode.FULL_LINEARIZATION:
                name += "Linearization"
            else:
                if include_result:
                    name += "ValueAnd"
                if len(which_args) == 1:
                    name += "Jacobian{}".format(which_args[0])
                elif len(which_args) == len(input_args):
                    name += "Jacobians"
                else:
                    name += "Jacobians{}".format("".join(str(s) for s in which_args))

        return Codegen(
            name=name,
            inputs=self.inputs,
            outputs=outputs,
            mode=self.mode,
            return_key=return_key,
            scalar_type=self.scalar_type,
            docstring="\n".join(docstring_lines),
        )
