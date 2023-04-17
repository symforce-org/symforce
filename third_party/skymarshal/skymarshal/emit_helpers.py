# aclint: py2 py3
# mypy: allow-untyped-defs
from __future__ import absolute_import

import io
import os
import typing as T

import jinja2
from skymarshal.syntax_tree import ConstMember


class TemplateRenderer(object):
    "A wrapper around a jinja template Environment instantiated for this package"

    def __init__(self, module_dir):
        template_dir = os.path.join(module_dir, "templates")
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir), lstrip_blocks=True, trim_blocks=True
        )

    def __call__(self, template_name, **kwargs):
        "Render the given template when called"
        return self.env.get_template(template_name).render(**kwargs)


class Code(io.StringIO):
    """A file-like object with methods for easy code creation.
    Modeled off the emit functions from the original lcmgen."""

    def start(self, indent, fmt, *args):
        self.write(u"    " * indent)
        self.add(fmt, *args)

    def add(self, fmt, *args):
        self.write(self._format(fmt, args))

    def end(self, fmt, *args):
        self.add(fmt, *args)
        self.write(u"\n")

    def _format(self, fmt, args):
        if isinstance(fmt, bytes):
            fmt = fmt.decode("utf-8")
        return fmt % args

    def __call__(self, indent, fmt, *args):
        self.write(u"    " * indent)
        self.write(self._format(fmt, args))
        self.write(u"\n")


class BaseBuilder(object):
    def __init__(self, package, name, full_name, comments, args):
        self.package = package
        self.name = name
        self._full_name = full_name
        self._comments = comments
        self.args = args

    @property
    def full_name(self):
        if self.args.package_prefix:
            return self.args.package_prefix + "." + self._full_name
        return self._full_name


class StructBuilder(BaseBuilder):
    """Helper class for converting a lcm struct into a destination code file."""

    def __init__(self, package, struct, args):
        super(StructBuilder, self).__init__(
            package, struct.name, struct.full_name, struct.comments, args
        )
        self.struct = struct
        self.members = [
            member for member in struct.members if not isinstance(member, ConstMember)
        ]  # type: T.List[T.Any]
        self.constants = [
            member for member in struct.members if isinstance(member, ConstMember)
        ]  # type: T.List[T.Any]

    @property
    def hash(self):
        return self.struct.compute_hash()


class EnumBuilder(BaseBuilder):
    """Helper class for converting a 'lcm' enum into a destination code file."""

    def __init__(self, package, enum, args):
        super(EnumBuilder, self).__init__(package, enum.name, enum.full_name, enum.comments, args)
        self.enum = enum
        self.cases = enum.cases
        self.num_cases = len(enum.cases)
        self.storage_type = enum.storage_type_ref

    @property
    def hash(self):
        return self.enum.compute_hash()

    # TODO(matt): There are probably more shared methods and common patterns.


render = TemplateRenderer(os.path.dirname(__file__))
