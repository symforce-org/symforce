# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.symbolic as sf
from symforce import path_util
from symforce import python_util
from symforce.codegen import RenderTemplateConfig
from symforce.codegen import template_util
from symforce.test_util import TestCase


class SymforcePybindCodegenTest(TestCase):
    def test_generate_pybind(self) -> None:
        output_dir = self.make_output_dir()

        for template_name in ("sym_type_casters.h.jinja", "sym_type_casters.cc.jinja"):
            output_name = template_name.replace(".jinja", "")
            template_util.render_template(
                template_dir=template_util.PYBIND_TEMPLATE_DIR,
                template_path=template_name,
                data=dict(
                    types=[cls.__name__ for cls in sf.GEO_TYPES + sf.CAM_TYPES],
                    python_util=python_util,
                ),
                config=RenderTemplateConfig(),
                output_path=output_dir / output_name,
            )

            self.compare_or_update_file(
                new_file=output_dir / output_name,
                path=path_util.symforce_data_root() / "symforce" / "pybind" / output_name,
            )


if __name__ == "__main__":
    TestCase.main()
