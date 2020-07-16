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
            scalar_type: Type used for generated scalar expressions
            docstring: The docstring to be used with the generated function
        """

        self.name = name

        assert isinstance(inputs, Values)
        assert isinstance(outputs, Values)
        self.inputs = inputs
        self.outputs = outputs

        self.mode = mode
        self.scalar_type = scalar_type

        if return_key is not None:
            assert return_key in outputs
        self.return_key = return_key

        self.docstring = docstring or Codegen.default_docstring(inputs=inputs, outputs=outputs)

        # TODO(nathan): Consider moving into a different function so that we can generate code separately
        self.intermediate_terms, self.output_terms = codegen_util.print_code(
            inputs=self.inputs, outputs=self.outputs, mode=self.mode
        )

    @classmethod
    def function(
        cls,
        name,  # type: str
        func,  # type: T.Callable
        input_types,  # type: T.Sequence[T.Type]
        mode,  # type: codegen_util.CodegenMode
        output_names=None,  # type: T.Sequence[str]
        return_key="",  # type: str
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
            assert output_names is not None
            if return_key:
                # If a return key is given, it must be valid (i.e. in output_names)
                assert return_key in output_names
        else:
            # Function returns single object
            output_terms = (res,)
            if output_names is None:
                output_names = ["res"]
            if return_key == "":
                # NOTE: We allow return_key to be None, which would signify that the object
                # should be returned using a pointer arugment
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

        def is_symbolic(T):
            # type: (T.Any) -> bool
            return isinstance(T, (sm.Expr, sm.Symbol))

        data["is_symbolic"] = is_symbolic
        data["issubclass"] = issubclass
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
        for d in (self.inputs, self.outputs):
            for key, value in d.items():
                if isinstance(value, Values):
                    types_to_generate.append((key, value))

        # Generate types from the Values objects in our inputs and outputs
        values_indices = {name: gen_type.index() for name, gen_type in types_to_generate}
        types_codegen_data = types_package_codegen.generate_types(
            package_name=namespace,
            values_indices=values_indices,
            shared_types=shared_types,
            mode=self.mode,
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

        # Generate the function
        if self.mode == codegen_util.CodegenMode.PYTHON2:
            logger.info('Creating python function "{}" at "{}"'.format(self.name, output_dir))
            templates.add(
                os.path.join(template_util.PYTHON_TEMPLATE_DIR, "function", "FUNCTION.py.jinja"),
                os.path.join(output_dir, self.name + ".py"),
                dict(self.common_data(), spec=self),
            )
            templates.add(
                os.path.join(template_util.PYTHON_TEMPLATE_DIR, "function", "__init__.py.jinja"),
                os.path.join(output_dir, "__init__.py"),
                dict(self.common_data(), spec=self),
            )
        elif self.mode == codegen_util.CodegenMode.CPP:
            logger.info('Creating C++ function "{}" at "{}"'.format(self.name, output_dir))
            templates.add(
                os.path.join(template_util.CPP_TEMPLATE_DIR, "function", "FUNCTION.h.jinja"),
                os.path.join(output_dir, python_util.camelcase_to_snakecase(self.name) + ".h"),
                dict(self.common_data(), spec=self),
            )
        else:
            raise NotImplementedError('Unknown mode: "{}"'.format(self.mode))

        templates.render()

        return {
            "generated_files": [v[1] for v in templates.items],
            "output_dir": output_dir,
        }

    @staticmethod
    def default_docstring(inputs, outputs, original_function=None):
        # type: (Values, Values, T.Callable) -> str
        """
        Create a default docstring if no other is available from the function or caller.
        """
        # If the function is an instance method, remove the type associated with the class
        input_types = [
            ops.StorageOps.get_type(arg).__name__ for name, arg in inputs.items() if name != "self"
        ]
        output_types = [ops.StorageOps.get_type(arg).__name__ for arg in outputs.values()]

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
