import os
import numpy as np
import tempfile

from symforce import logger
from symforce import types as T
from symforce.values import Values
from symforce.codegen import template_util
from symforce.codegen import codegen_util


def generate_types(
    package_name,  # type: str
    values_indices,  # type: T.Mapping[str, T.Dict[str, T.Any]]
    shared_types=None,  # type: T.Mapping[str, str]
    mode=codegen_util.CodegenMode.PYTHON2,  # type: codegen_util.CodegenMode
    scalar_type="double",  # type: str
    output_dir=None,  # type: T.Optional[str]
    templates=None,  # type: template_util.TemplateList
):
    # type: (...) -> T.Dict[str, T.Any]
    """
    Generate a package with type structs.
    """
    # TODO(nathan): I feel like using shared_types for both types shared within the package as well
    # as for external types is really confusing (I've spent far too long trying to figure out exactly
    # what `shared_types` means/does).
    # Create output directory if needed
    if output_dir is None:
        output_dir = tempfile.mkdtemp(
            prefix="sf_codegen_types_{}_".format(package_name), dir="/tmp"
        )
        logger.debug("Creating temp directory: {}".format(output_dir))
    package_dir = os.path.join(output_dir, package_name)

    using_external_templates = True
    if templates is None:
        # List of (template_path, output_path, data)
        templates = template_util.TemplateList()
        using_external_templates = False

    types_dict = build_types_dict(
        package_name=package_name, values_indices=values_indices, shared_types=shared_types
    )

    # Default data for templates
    data = {
        "name": package_name,
        "scalar_type": scalar_type,
        "types_dict": types_dict,
    }

    if mode == codegen_util.CodegenMode.PYTHON2:
        logger.info('Creating Python types package at: "{}"'.format(package_dir))
        template_dir = os.path.join(template_util.PYTHON_TEMPLATE_DIR, "types_package")

        data["np_scalar_types"] = codegen_util.NUMPY_DTYPE_FROM_SCALAR_TYPE

        for typename in types_dict:
            # If a module is specified, this type is external - don't generate it
            if "." in typename:
                continue

            # Type definition
            templates.add(
                os.path.join(template_dir, "type.py.jinja"),
                os.path.join(package_dir, "{}.py".format(typename)),
                dict(data, typename=typename),
            )

            # Storage ops
            templates.add(
                os.path.join(template_dir, "storage_ops", "storage_ops_type.py.jinja"),
                os.path.join(package_dir, "storage_ops", "{}.py".format(typename)),
                dict(data, typename=typename),
            )

        # Init that imports all types
        templates.add(
            os.path.join(template_dir, "__init__.py.jinja"),
            os.path.join(package_dir, "__init__.py"),
            dict(data, typenames=types_dict.keys()),
        )

        # Init that contains a registration function to add ops to types
        templates.add(
            os.path.join(template_dir, "storage_ops", "__init__.py.jinja"),
            os.path.join(package_dir, "storage_ops", "__init__.py"),
            dict(data, typenames=types_dict.keys()),
        )

    elif mode == codegen_util.CodegenMode.CPP:
        logger.info('Creating C++ types package at: "{}"'.format(package_dir))
        template_dir = os.path.join(template_util.CPP_TEMPLATE_DIR, "types_package")

        for typename in types_dict:
            # If a module is specified, this type is external - don't generate it
            if "." in typename:
                continue

            # Type definition
            templates.add(
                os.path.join(template_dir, "type.h.jinja"),
                os.path.join(package_dir, "{}.h".format(typename)),
                dict(data, typename=typename),
            )

            # Storage ops
            templates.add(
                os.path.join(template_dir, "storage_ops/storage_ops_type.h.jinja"),
                os.path.join(package_dir, "storage_ops/{}.h".format(typename)),
                dict(data, typename=typename),
            )

        # Init that contains traits definition and imports all types
        storage_ops_file = os.path.join(package_dir, "storage_ops.h")
        templates.add(
            os.path.join(template_dir, "storage_ops/storage_ops.h.jinja"),
            os.path.join(package_dir, "storage_ops.h"),
            dict(data, typenames=types_dict.keys()),
        )

    else:
        raise NotImplementedError('Unknown mode: "{}"'.format(mode))

    if not using_external_templates:
        templates.render()

    # Save input args for handy reference
    codegen_data = {}  # type: T.Dict[str, T.Any]
    codegen_data["package_name"] = package_name
    codegen_data["values_indices"] = values_indices
    codegen_data["shared_types"] = shared_types
    codegen_data["mode"] = mode
    codegen_data["scalar_type"] = scalar_type

    # Save outputs and intermediates
    codegen_data["output_dir"] = output_dir
    codegen_data["package_dir"] = package_dir
    codegen_data["types_dict"] = types_dict

    # TODO(nathan): This doesn't include subtypes yet
    codegen_data["typenames_dict"] = dict()  # Maps typenames to generated types
    codegen_data["namespaces_dict"] = dict()  # Maps typenames to namespaces
    for name in values_indices.keys():
        for typename, data in types_dict.items():
            if name == data["unformatted_typename"]:
                codegen_data["typenames_dict"][name] = typename.split(".")[-1]
                if shared_types is not None and name in shared_types and "." in shared_types[name]:
                    codegen_data["namespaces_dict"][name] = shared_types[name].split(".")[0]
                else:
                    codegen_data["namespaces_dict"][name] = package_name

    return codegen_data


def build_types_dict(
    package_name,  # type: str
    values_indices,  # type: T.Mapping[str, T.Dict[str, T.Any]]
    shared_types=None,  # type: T.Mapping[str, str]
):
    # type: (...) -> T.Dict[str, T.Dict[str, T.Any]]
    """
    Compute the structure of the types we need to generate for the given Values.
    """
    if shared_types is None:
        shared_types = dict()

    types_dict = dict()  # type: T.Dict[str, T.Dict[str, T.Any]]

    for key, index in values_indices.items():
        _fill_types_dict_recursive(
            key=key,
            index=index,
            package_name=package_name,
            shared_types=shared_types,
            types_dict=types_dict,
        )

    return types_dict


def typename_from_key(key, shared_types):
    # type: (str, T.Mapping[str, str]) -> str
    """
    Compute a typename from a key, or from shared_types if provided by the user.
    """
    return shared_types.get(key, key.replace(".", "_") + "_t")


def _fill_types_dict_recursive(
    key,  # type: str
    index,  # type: T.Dict
    package_name,  # type: str
    shared_types,  # type: T.Mapping[str, str]
    types_dict,  # type: T.Dict[str, T.Dict[str, T.Any]]
):
    # type: (...) -> None
    """
    Recursively compute type information from the key and values index and fill into types_dict.
    """
    data = {}  # type: T.Dict[str, T.Any]

    typename = typename_from_key(key, shared_types)
    data["typename"] = typename
    data["unformatted_typename"] = key

    # Add the current module for cases where it's not specified
    data["full_typename"] = typename if "." in typename else ".".join([package_name, typename])

    data["index"] = index
    data["keys_recursive"] = Values.scalar_keys_recursive_from_index(index)
    data["storage_dims"] = {key: np.prod(info[2]) for key, info in index.items()}

    # Process child types
    data["subtypes"] = {}
    for subkey, (_, datatype, _, item_index) in index.items():
        if not datatype == "Values":
            continue

        full_subkey = "{}.{}".format(key, subkey)
        data["subtypes"][subkey] = typename_from_key(full_subkey, shared_types)

        _fill_types_dict_recursive(
            key=full_subkey,
            index=item_index,
            package_name=package_name,
            shared_types=shared_types,
            types_dict=types_dict,
        )

    if typename in types_dict:
        assert set(types_dict[typename]["keys_recursive"]) == set(data["keys_recursive"])
        assert set(types_dict[typename]["full_typename"]) == set(data["full_typename"])

    types_dict[typename] = data
