import os

from symforce import typing as T
from symforce.codegen import Codegen, template_util
from symforce.values import generated_key_selection
from symforce.values.values import Values


def generate_values_keys(
    values: Values,
    output_dir: str,
    namespace: str = "sym",
    generated_file_name: str = "keys.h",
    excluded_keys: T.Set[generated_key_selection.GeneratedKey] = None,
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
    """
    items = values.items_recursive()
    keys = [item[0] for item in items]
    generated_keys = generated_key_selection.pick_generated_keys_for_variable_names(
        keys, excluded_keys
    )
    vars_to_generate = [(key, generated_keys[key], value) for key, value in items]

    cpp_function_dir = os.path.join(output_dir, "cpp", "symforce", namespace)
    template_util.render_template(
        template_path=os.path.join(template_util.CPP_TEMPLATE_DIR, "keys.h.jinja"),
        data=dict(Codegen.common_data(), namespace=namespace, vars=vars_to_generate),
        output_path=os.path.join(cpp_function_dir, generated_file_name),
    )
