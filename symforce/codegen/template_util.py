import collections
import jinja2
import os

from symforce import logger
from symforce import types as T
from symforce.codegen import format_util

CURRENT_DIR = os.path.dirname(__file__)
CPP_TEMPLATE_DIR = os.path.join(CURRENT_DIR, "cpp_templates")
PYTHON_TEMPLATE_DIR = os.path.join(CURRENT_DIR, "python_templates")
LCM_TEMPLATE_DIR = os.path.join(CURRENT_DIR, "lcm_templates")


class RelEnvironment(jinja2.Environment):
    """
    Override join_path() to enable relative template paths. Modified from the below post.

    https://stackoverflow.com/questions/8512677/how-to-include-a-template-with-relative-path-in-jinja2
    """

    def join_path(self, template: T.Union[jinja2.Template, str], parent: str) -> str:
        return os.path.normpath(os.path.join(os.path.dirname(parent), str(template)))


def render_template(
    template_path: str,
    data: T.Dict[str, T.Any],
    output_path: T.Optional[str] = None,
    template_dir: str = CURRENT_DIR,
) -> str:
    """
    Boiler plate to render template. Returns the rendered string and optionally writes to file.

    Args:
        template_path: file path of the template to render
        data: dictionary of inputs for template
        output_path: If provided, writes to file
        template_dir: Base directory where templates are found, defaults to symforce/codegen
    """
    logger.debug(f"Template  IN <-- {template_path}")
    if output_path:
        logger.debug(f"Template OUT --> {output_path}")

    template_name = os.path.relpath(template_path, template_dir)

    loader = jinja2.FileSystemLoader(template_dir)
    env = RelEnvironment(
        loader=loader,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )

    template = env.get_template(template_name)
    rendered_str = str(template.render(**data))

    if not template_path.endswith(".jinja"):
        raise ValueError("template must be of the form path/to/file.ext.jinja")
    template_filename_without_jinja = os.path.basename(template_path)[: -len(".jinja")]
    extension = template_filename_without_jinja.split(".")[-1]
    if extension in ("c", "cpp", "cxx", "cc", "tcc", "h", "hpp", "hxx", "hh", "cu", "cuh"):
        # Come up with a fake filename to give to the formatter just for formatting purposes, even
        # if this isn't being written to disk
        if output_path is not None:
            format_cpp_filename = os.path.basename(output_path)
        else:
            format_cpp_filename = template_filename_without_jinja

        rendered_str = format_util.format_cpp(
            rendered_str, filename=os.path.join(CURRENT_DIR, format_cpp_filename)
        )
    elif extension == "py":
        rendered_str = format_util.format_py(rendered_str)

    if output_path:
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_path, "w") as f:
            f.write(rendered_str)

    return rendered_str


class TemplateList:
    """
    Helper class to keep a list of (template_path, output_path, data) and render
    all templates in one go.
    """

    Entry = collections.namedtuple("TemplateListEntry", ["template_path", "output_path", "data"])

    def __init__(self) -> None:
        self.items: T.List = []

    def add(self, template_path: str, output_path: str, data: T.Dict[str, T.Any]) -> None:
        self.items.append(
            self.Entry(template_path=template_path, output_path=output_path, data=data)
        )

    def render(self) -> None:
        for entry in self.items:
            render_template(
                template_path=entry.template_path, output_path=entry.output_path, data=entry.data
            )
