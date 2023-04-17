# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()

from symforce import path_util
from symforce.codegen import RenderTemplateConfig
from symforce.codegen import template_util
from symforce.codegen.lcm_types_codegen import lcm_symforce_types_data
from symforce.test_util import TestCase


class SymforceLcmCodegenTest(TestCase):
    def test_generate_lcm(self) -> None:
        output_dir = self.make_output_dir("sf_lcm_codegen_test")

        template_util.render_template(
            template_dir=template_util.LCM_TEMPLATE_DIR,
            template_path="symforce_types.lcm.jinja",
            data=lcm_symforce_types_data(),
            config=RenderTemplateConfig(),
            output_path=output_dir / "symforce_types.lcm",
        )

        self.compare_or_update_file(
            new_file=output_dir / "symforce_types.lcm",
            path=path_util.symforce_data_root() / "lcmtypes" / "symforce_types.lcm",
        )


if __name__ == "__main__":
    TestCase.main()
