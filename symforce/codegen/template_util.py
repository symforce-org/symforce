from __future__ import absolute_import

import collections
import jinja2
import os

from symforce import logger

CURRENT_DIR = os.path.dirname(__file__)
CPP_TEMPLATE_DIR = os.path.join(CURRENT_DIR, "cpp", "templates")
PYTHON_TEMPLATE_DIR = os.path.join(CURRENT_DIR, "python", "templates")


class RelEnvironment(jinja2.Environment):
    """
    Override join_path() to enable relative template paths. Modified from the below post.

    https://stackoverflow.com/questions/8512677/how-to-include-a-template-with-relative-path-in-jinja2
    """

    def join_path(self, template, parent):
        return os.path.normpath(os.path.join(os.path.dirname(parent), template))


def render_template(template_path, data, output_path=None):
    """
    Boiler plate to render template. Returns the rendered string and optionally writes to file.

    Args:
        template_path (str): file path of the template to render
        data (dict): dictionary of inputs for template
        output_path (str): If provided, writes to file

    Returns:
        (str): rendered template
    """
    logger.debug("Template  IN <-- {}".format(template_path))
    if output_path:
        logger.debug("Template OUT --> {}".format(output_path))

    template_dir = CURRENT_DIR
    template_name = os.path.relpath(template_path, template_dir)

    loader = jinja2.FileSystemLoader(template_dir)
    env = RelEnvironment(
        loader=loader, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
    )

    template = env.get_template(template_name)
    rendered_str = template.render(**data)

    if output_path:
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(output_path, "w") as f:
            f.write(rendered_str)

    return rendered_str


class TemplateList(object):
    """
    Helper class to keep a list of (template_path, output_path, data) and render
    all templates in one go.
    """

    Entry = collections.namedtuple("TemplateListEntry", ["template_path", "output_path", "data"])

    def __init__(self):
        self.items = []

    def add(self, template_path, output_path, data):
        self.items.append(
            self.Entry(template_path=template_path, output_path=output_path, data=data)
        )

    def render(self):
        for entry in self.items:
            render_template(
                template_path=entry.template_path, output_path=entry.output_path, data=entry.data
            )
