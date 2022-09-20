# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import re
from pathlib import Path

from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen import CppConfig
from symforce.codegen import template_util
from symforce.values import generated_key_selection
from symforce.values.values import Values


def generate_values_keys(
    values: Values,
    output_dir: T.Openable,
    namespace: str = "sym",
    generated_file_name: str = "keys.h",
    excluded_keys: T.Set[generated_key_selection.GeneratedKey] = None,
    skip_directory_nesting: bool = False,
) -> None:
    """
    Generate C++ variables to easily create `sym::Key`s from the python key names

    Args:
        values: Will generate an entry for each (recursive) key in the values
        output_dir: Directory in which to output the generated header
        namespace: Namepace for the generated header
        generated_file_name: Filename of the generated header
        excluded_keys: Set of disallowed generated keys (for instance, if that key is used
                       elsewhere)
        skip_directory_nesting: Generate the output file directly into output_dir instead of adding
                                the usual directory structure inside output_dir
    """
    config = CppConfig()
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    items = values.items_recursive()
    items = list({re.sub(r"(\[[0-9]+\])+$", "", key): value for key, value in items}.items())
    keys = [item[0] for item in items]
    generated_keys = generated_key_selection.pick_generated_keys_for_variable_names(
        keys, excluded_keys
    )
    vars_to_generate = [(key, generated_keys[key], value) for key, value in items]

    if skip_directory_nesting:
        cpp_function_dir = output_dir
    else:
        cpp_function_dir = output_dir / "cpp" / "symforce" / namespace

    template_util.render_template(
        template_dir=config.template_dir(),
        template_path="keys.h.jinja",
        data=dict(Codegen.common_data(), namespace=namespace, vars=vars_to_generate),
        output_path=cpp_function_dir / generated_file_name,
    )
