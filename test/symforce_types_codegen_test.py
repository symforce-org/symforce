import logging
import os

from symforce import geo
from symforce import logger
from symforce import python_util
from symforce import sympy as sm
from symforce import types as T
from symforce.codegen import types_package_codegen
from symforce.codegen import CodegenMode
from symforce.codegen import codegen_util
from symforce.test_util import TestCase
from symforce.values import Values


class SymforceTypesCodegenTest(TestCase):
    """
    Test symforce.codegen.types_package_codegen
    """

    @staticmethod
    def build_values():
        # type: () -> T.Tuple[Values, Values]
        """
        Create some example input/output values.
        """
        inputs = Values()
        x, y = sm.symbols("x y")
        inputs.add(x)
        inputs.add(y)

        inputs["rot"] = geo.Rot3().symbolic("rot")

        # Scalar
        inputs.add(sm.Symbol("constants.epsilon"))

        # Vector
        sub_values = inputs.copy()
        inputs["values_vec"] = [sub_values, sub_values]

        with inputs.scope("states"):
            # Array element, turns into std::array
            inputs["p"] = geo.V2.symbolic("p")

        outputs = Values()
        outputs["foo"] = x ** 2 + inputs["rot"].q.w
        outputs["bar"] = inputs.attr.constants.epsilon + sm.sin(inputs.attr.y) + x ** 2

        return inputs, outputs

    def types_codegen_helper(
        self,
        name,  # type: str
        scalar_type,  # type: str
        shared_types,  # type: T.Mapping[str, str]
        expected_types,  # type: T.Sequence[str]
    ):
        # type: (...) -> T.Dict[str, T.Any]
        """
        Helper to test generation with less duplicated code.
        """
        inputs, outputs = self.build_values()

        codegen_data = types_package_codegen.generate_types(
            package_name=name,
            values_indices=dict(input=inputs.index(), output=outputs.index()),
            shared_types=shared_types,
            scalar_type=scalar_type,
        )

        types_dict = codegen_data["types_dict"]

        # Check types dict
        self.assertSetEqual(set(types_dict.keys()), set(expected_types))

        # Check generated files
        for typename in expected_types:
            python_gen_path = os.path.join(
                codegen_data["python_types_dir"], codegen_data["package_name"], typename + ".py",
            )
            cpp_gen_path = os.path.join(
                codegen_data["output_dir"],
                "cpp",
                "lcmtypes",
                codegen_data["package_name"],
                typename + ".hpp",
            )

            # If the typename has a module, it's assumed to be external and not generated.
            expecting_generated_type = "." not in typename
            self.assertEqual(os.path.isfile(python_gen_path), expecting_generated_type)
            self.assertEqual(os.path.isfile(cpp_gen_path), expecting_generated_type)

        return codegen_data

    def test_types_codegen_vanilla(self):
        # type: () -> None
        """
        No shared types.
        """
        codegen_data = self.types_codegen_helper(
            name="vanilla",
            scalar_type="double",
            shared_types={},
            expected_types=(
                "input_t",
                "output_t",
                "input_states_t",
                "input_constants_t",
                "input_values_vec_t",
                "input_values_vec_constants_t",
            ),
        )

        package_dir = os.path.join(codegen_data["python_types_dir"], codegen_data["package_name"])
        package = codegen_util.load_generated_package(package_dir)

        inp = package.input_t()
        inp.x = 1.2
        inp.y = 4.5
        inp.constants.epsilon = -2
        inp2 = package.input_t.decode(inp.encode())
        self.assertEqual(inp.x, inp2.x)
        self.assertEqual(inp.constants.epsilon, inp2.constants.epsilon)

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(codegen_data["output_dir"])

    def test_types_codegen_renames(self):
        # type: () -> None
        """
        Give some fields specific names, still within the module.
        """
        shared_types = {
            "input": "foo_t",
            "output": "bar_t",
            "input.states": "zoomba_t",
            "input.values_vec": "vec_t",
        }
        expected_types = (
            "foo_t",
            "bar_t",
            "zoomba_t",
            "input_constants_t",
            "vec_t",
            "input_values_vec_constants_t",
        )

        codegen_data = self.types_codegen_helper(
            name="renames",
            scalar_type="double",
            shared_types=shared_types,
            expected_types=expected_types,
        )

        package_dir = os.path.join(codegen_data["python_types_dir"], codegen_data["package_name"])
        package = codegen_util.load_generated_package(package_dir)

        inp = package.foo_t()
        inp.x = 1.2
        inp.y = 4.5
        inp.constants.epsilon = -2
        inp2 = package.foo_t.decode(inp.encode())
        self.assertEqual(inp.x, inp2.x)
        self.assertEqual(inp.constants.epsilon, inp2.constants.epsilon)

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(codegen_data["output_dir"])

    def test_types_codegen_external(self):
        # type: () -> None
        """
        Use external types for some fields, don't generate them.
        """
        shared_types = {
            "output": "other_module.bar_t",
            "input.states": "external.zoomba_t",
            "input.values_vec": "external.vec_t",
        }
        expected_types = (
            "input_t",
            "other_module.bar_t",
            "external.zoomba_t",
            "input_constants_t",
            "external.vec_t",
        )

        codegen_data = self.types_codegen_helper(
            name="external",
            scalar_type="double",
            shared_types=shared_types,
            expected_types=expected_types,
        )

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(codegen_data["output_dir"])

    def test_types_codegen_reuse(self):
        # type: () -> None
        """
        Use the same generated type multiple times within the input and output struct.
        """
        inputs = Values()
        inputs.add("foo")

        with inputs.scope("one"):
            inputs.add("id")
            inputs["R"] = geo.Rot3().symbolic("rot")

        with inputs.scope("two"):
            inputs.add("id")
            inputs["R"] = geo.Rot3().symbolic("rot")

        outputs = Values()
        outputs.add("id")
        outputs["R"] = geo.Rot3().symbolic("rot")

        shared_types = {"input.one": "rot_t", "input.two": "rot_t", "output": "rot_t"}

        codegen_data = types_package_codegen.generate_types(
            package_name="reuse",
            values_indices=dict(input=inputs.index(), output=outputs.index()),
            shared_types=shared_types,
            scalar_type="double",
        )

        types_dict = codegen_data["types_dict"]

        self.assertEqual(set(types_dict.keys()), set(["input_t", "rot_t"]))

        package_dir = os.path.join(codegen_data["python_types_dir"], codegen_data["package_name"])
        package = codegen_util.load_generated_package(package_dir)

        rot = package.rot_t()
        rot.id = 1
        rot.R = geo.Rot3.identity().to_storage()

        inp = package.input_t()
        inp.foo = 1.2
        inp.one = rot
        inp.two = rot

        inp2 = package.input_t.decode(inp.encode())
        self.assertNear(rot.R, inp2.two.R, places=9)

        # Clean up
        if logger.level != logging.DEBUG:
            python_util.remove_if_exists(codegen_data["output_dir"])


if __name__ == "__main__":
    TestCase.main()
