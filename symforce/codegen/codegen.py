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
from symforce.codegen import types_package_codegen

CURRENT_DIR = os.path.dirname(__file__)


class Codegen(object):
    """
    Class used for generating code from symbolic expressions or functions.

    Codegen objects can either be used to generate standalone functions, or
    as specifications in a larger code generation pipeline. Each codegen object
    defines an input/output relationship between a set of symbolic inputs and
    a set of symbolic output expressions written in terms of the inputs.
    """

    def __init__(
        self,
        name,  # type: str
        inputs,  # type: Values
        outputs,  # type: Values
        mode,  # type: codegen_util.CodegenMode
        return_key=None,  # type: str
        sparse_matrices=None,  # type: T.List[str]
        scalar_type="double",  # type: str
        docstring=None,  # type: str
    ):
        # type: (...) -> None
        """
        Creates the Codegen specification.

        Args:
            name: Name of the function to be generated
            inputs: Values object specifying names and symbolic inputs to the function
            outputs: Values object specifying names and output expressions (written in terms
                of the symbolic inputs) of the function
            mode: Programming language in which the function is to be generated
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
        assert all([python_util.is_valid_variable_name(k) for k in inputs.subkeys_recursive()])
        assert all([python_util.is_valid_variable_name(k) for k in outputs.subkeys_recursive()])

        # Symbols in inputs must be unique
        assert len(set(inputs.to_storage())) == len(
            inputs.to_storage()
        ), "Symbols in inputs must be unique. Duplicate symbols = {}".format(
            [symbol for symbol in inputs.to_storage() if inputs.to_storage().count(symbol) > 1]
        )

        # Outputs must not have same variable names/keys as inputs
        assert all([key not in outputs.keys() for key in inputs.keys()])

        self.inputs = inputs
        self.outputs = outputs

        self.mode = mode
        self.scalar_type = scalar_type

        if return_key is not None:
            assert return_key in outputs
        self.return_key = return_key

        # Mapping between sparse matrix keys and constants needed for static CSC construction
        self.sparse_mat_data = {}  # type: T.Dict[str, T.Dict[str, T.Any]]
        if sparse_matrices is not None:
            assert all([key in outputs for key in sparse_matrices])
            assert all([isinstance(outputs[key], geo.Matrix) for key in sparse_matrices])
            for key in sparse_matrices:
                self.sparse_mat_data[key] = codegen_util.get_sparse_mat_data(outputs[key])

        self.docstring = docstring or Codegen.default_docstring(inputs=inputs, outputs=outputs)

        # TODO(nathan): Consider moving into a different function so that we can generate code separately
        self.intermediate_terms, self.output_terms, self.sparse_terms = codegen_util.print_code(
            inputs=self.inputs,
            outputs=self.outputs,
            sparse_mat_data=self.sparse_mat_data,
            mode=self.mode,
        )

    @classmethod
    def function(
        cls,
        name,  # type: str
        func,  # type: T.Callable
        input_types,  # type: T.Sequence[T.Type]
        mode,  # type: codegen_util.CodegenMode
        output_names=None,  # type: T.Sequence[str]
        return_key=None,  # type: str
        docstring=None,  # type: str
    ):
        # type: (...)  -> Codegen
        """
        Creates a Codegen object from a symbolic python function.

        Args:
            name: Name of the function to be generated
            func: Python function
            input_types: List of types of the inputs to the given function
            mode: Programming language in which the function is to be generated
            output_names: Optional if only one object is returned by the function.
                If multiple objects are returned, they must be named.
            return_key: If multiple objects are returned, the generated function will return
                the object with this name (must be in output_names)
        """
        arg_spec = codegen_util.get_function_argspec(func)

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
    def common_data():
        # type: () -> T.Dict[str, T.Any]
        """
        Return common template data for code generation.
        """
        data = {}  # type: T.Dict[str, T.Any]
        data["ops"] = ops
        data["Symbol"] = sm.Symbol
        data["Matrix"] = geo.Matrix
        data["Values"] = Values
        data["scalar_types"] = ("double", "float")
        data["camelcase_to_snakecase"] = python_util.camelcase_to_snakecase
        data["python_util"] = python_util

        def is_symbolic(T):
            # type: (T.Any) -> bool
            return isinstance(T, (sm.Expr, sm.Symbol))

        data["is_symbolic"] = is_symbolic
        data["issubclass"] = issubclass
        data["is_sequence"] = lambda arg: isinstance(arg, (list, tuple))
        return data

    def generate_function(
        self,
        output_dir=None,  # type: str
        shared_types=None,  # type: T.Mapping[str, str]
        namespace="sym",  # type: str
    ):
        # type: (...) -> T.Dict[str, T.Any]
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
            shared_types: Mapping between types defined as part of this codegen object (e.g. keys in
                self.inputs that map to Values objects) and previously generated external types.
            namespace: Namespace for the generated function and any generated types.
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="sf_codegen_{}_".format(self.name), dir="/tmp")
            logger.debug("Creating temp directory: {}".format(output_dir))

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
            values_indices=values_indices,
            shared_types=shared_types,
            scalar_type=self.scalar_type,
            output_dir=output_dir,
            templates=templates,
        )

        # Maps typenames to generated types
        self.typenames_dict = types_codegen_data["typenames_dict"]
        # Maps typenames to namespaces
        self.namespaces_dict = types_codegen_data["namespaces_dict"]
        self.unique_namespaces = set(v for v in self.namespaces_dict.values())

        # Namespace of this function + generated types
        self.namespace = namespace

        output_data = {
            "output_dir": output_dir,
            "lcm_type_dir": types_codegen_data["lcm_type_dir"],
        }

        # Generate the function
        if self.mode == codegen_util.CodegenMode.PYTHON2:
            python_function_dir = os.path.join(output_dir, "python2.7", "symforce", namespace)
            logger.info(
                'Creating python function "{}" at "{}"'.format(self.name, python_function_dir)
            )
            templates.add(
                os.path.join(template_util.PYTHON_TEMPLATE_DIR, "function", "FUNCTION.py.jinja"),
                os.path.join(python_function_dir, self.name + ".py"),
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
            logger.info('Creating C++ function "{}" at "{}"'.format(self.name, cpp_function_dir))

            templates.add(
                os.path.join(template_util.CPP_TEMPLATE_DIR, "function", "FUNCTION.h.jinja"),
                os.path.join(
                    cpp_function_dir, python_util.camelcase_to_snakecase(self.name) + ".h"
                ),
                dict(self.common_data(), spec=self),
            )

            output_data["cpp_function_dir"] = cpp_function_dir
        else:
            raise NotImplementedError('Unknown mode: "{}"'.format(self.mode))

        templates.render()
        lcm_data = codegen_util.generate_lcm_types(
            lcm_type_dir=types_codegen_data["lcm_type_dir"],
            typenames=types_codegen_data["types_dict"].keys(),
        )
        output_data.update(lcm_data)

        output_data["generated_files"] = [v[1] for v in templates.items]

        return output_data

    @staticmethod
    def default_docstring(inputs, outputs, original_function=None):
        # type: (Values, Values, T.Callable) -> str
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
