# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import copy

import symforce

symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from symforce import typing as T
from symforce.codegen import codegen_util
from symforce.codegen import types_package_codegen
from symforce.test_util import TestCase
from symforce.values import Values


class SymforceTypesCodegenTest(TestCase):
    """
    Test symforce.codegen.types_package_codegen
    """

    @staticmethod
    def build_values() -> T.Tuple[Values, Values]:
        """
        Create some example input/output values.
        """
        inputs = Values()
        x, y = sf.symbols("x y")
        inputs.add(x)
        inputs.add(y)

        inputs["rot"] = sf.Rot3().symbolic("rot")

        # Scalar
        inputs.add(sf.Symbol("constants.epsilon"))

        # Vector
        sub_values = copy.deepcopy(inputs)
        inputs["values_vec"] = [sub_values, sub_values]

        with inputs.scope("states"):
            # Array element, turns into std::array
            inputs["p"] = sf.V2.symbolic("p")

        outputs = Values()
        outputs["foo"] = x**2 + inputs["rot"].q.w
        outputs["bar"] = inputs.attr.constants.epsilon + sf.sin(inputs.attr.y) + x**2

        return inputs, outputs

    def types_codegen_helper(
        self,
        name: str,
        scalar_type: str,
        shared_types: T.Mapping[str, str],
        expected_types: T.Sequence[str],
    ) -> types_package_codegen.TypesCodegenData:
        """
        Helper to test generation with less duplicated code.
        """
        inputs, outputs = self.build_values()

        output_dir = self.make_output_dir("sf_types_codegen_")
        codegen_data = types_package_codegen.generate_types(
            package_name=name,
            file_name=name,
            values_indices=dict(input=inputs.index(), output=outputs.index()),
            use_eigen_types=True,
            shared_types=shared_types,
            scalar_type=scalar_type,
            output_dir=output_dir,
        )

        types_dict = codegen_data.types_dict

        # Check types dict
        self.assertSetEqual(set(types_dict.keys()), set(expected_types))

        # Check generated files
        for typename in expected_types:
            assert codegen_data.lcm_bindings_dirs is not None
            python_gen_path = (
                codegen_data.lcm_bindings_dirs.python_types_dir
                / "lcmtypes"
                / codegen_data.package_name
                / f"_{typename}.py"
            )
            cpp_gen_path = (
                codegen_data.output_dir
                / "cpp"
                / "lcmtypes"
                / codegen_data.package_name
                / (typename + ".hpp")
            )

            # If the typename has a module, it's assumed to be external and not generated.
            expecting_generated_type = "." not in typename
            self.assertEqual(python_gen_path.is_file(), expecting_generated_type)
            self.assertEqual(cpp_gen_path.is_file(), expecting_generated_type)

        return codegen_data

    def test_types_codegen_vanilla(self) -> None:
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

        assert codegen_data.lcm_bindings_dirs is not None
        input_t = codegen_util.load_generated_lcmtype(
            "vanilla", "input_t", codegen_data.lcm_bindings_dirs.python_types_dir
        )

        inp = input_t()
        inp.x = 1.2
        inp.y = 4.5
        inp.constants.epsilon = -2
        inp2 = input_t.decode(inp.encode())
        self.assertEqual(inp.x, inp2.x)
        self.assertEqual(inp.constants.epsilon, inp2.constants.epsilon)

    def test_types_codegen_renames(self) -> None:
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

        assert codegen_data.lcm_bindings_dirs is not None
        foo_t = codegen_util.load_generated_lcmtype(
            "renames", "foo_t", codegen_data.lcm_bindings_dirs.python_types_dir
        )

        inp = foo_t()
        inp.x = 1.2
        inp.y = 4.5
        inp.constants.epsilon = -2
        inp2 = foo_t.decode(inp.encode())
        self.assertEqual(inp.x, inp2.x)
        self.assertEqual(inp.constants.epsilon, inp2.constants.epsilon)

    def test_types_codegen_external(self) -> None:
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

        self.types_codegen_helper(
            name="external",
            scalar_type="double",
            shared_types=shared_types,
            expected_types=expected_types,
        )

    def test_types_codegen_reuse(self) -> None:
        """
        Use the same generated type multiple times within the input and output struct.
        """
        inputs = Values()
        inputs.add("foo")

        with inputs.scope("one"):
            inputs.add("id")
            inputs["R"] = sf.Rot3().symbolic("rot")

        with inputs.scope("two"):
            inputs.add("id")
            inputs["R"] = sf.Rot3().symbolic("rot")

        outputs = Values()
        outputs.add("id")
        outputs["R"] = sf.Rot3().symbolic("rot")

        shared_types = {"input.one": "rot_t", "input.two": "rot_t", "output": "rot_t"}

        output_dir = self.make_output_dir("sf_types_codegen_reuse_")
        codegen_data = types_package_codegen.generate_types(
            package_name="reuse",
            file_name="reuse",
            values_indices=dict(input=inputs.index(), output=outputs.index()),
            use_eigen_types=True,
            shared_types=shared_types,
            scalar_type="double",
            output_dir=output_dir,
        )

        types_dict = codegen_data.types_dict

        self.assertEqual(set(types_dict.keys()), {"input_t", "rot_t"})

        assert codegen_data.lcm_bindings_dirs is not None
        rot_t = codegen_util.load_generated_lcmtype(
            "reuse", "rot_t", codegen_data.lcm_bindings_dirs.python_types_dir
        )

        input_t = codegen_util.load_generated_lcmtype(
            "reuse", "input_t", codegen_data.lcm_bindings_dirs.python_types_dir
        )

        rot = rot_t()
        rot.id = 1
        rot.R.data = sf.Rot3.identity().to_storage()

        inp = input_t()
        inp.foo = 1.2
        inp.one = rot
        inp.two = rot

        inp2 = input_t.decode(inp.encode())
        self.assertStorageNear(rot.R.data, inp2.two.R.data, places=9)


if __name__ == "__main__":
    TestCase.main()
