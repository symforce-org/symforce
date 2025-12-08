# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import multiprocessing
import os
import re
import subprocess
import sys
import typing as T
from pathlib import Path

from setuptools import Extension
from setuptools import find_namespace_packages
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install

SOURCE_DIR = Path(__file__).resolve().parent
ESCAPED_SOURCE_DIR = Path(str(SOURCE_DIR).replace(" ", "%20"))


class CMakeExtension(Extension):
    """
    CMake extension type.
    """

    def __init__(self, name: str):
        Extension.__init__(self, name, sources=[])


class PatchDevelop(develop):
    """
    develop is the legacy command (pre setuptools==64.0.0, which implemented
    the pep 660 hook build_editable) to build a package in editable mode.

    This will be used by setuptools < 64.0.0, and not be used by
    setuptools >= 64.0.0 during a normal `pip install -e .` run (of course,
    you could still run `python setup.py develop` to run it with any version
    of setuptools).
    """

    def run(self) -> None:  # type: ignore[override, unused-ignore]
        # NOTE(brad): In setuptools 64.0.0, the editable_mode field of the
        # build_ext was added and is set to True during a pep 660 editable
        # install.
        # Since we want to use this field in our CMakeBuild extension of
        # build_ext to identify if we're performing an editable install
        # regardless of whether we're building with develop or build_editable,
        # (both before and after version 64.0.0), we patch develop to add this
        # field to build_ext and set it to True.
        self.distribution.get_command_obj("build_ext").editable_mode = True  # type: ignore[misc]
        super().run()


class CMakeBuild(build_ext):
    """
    Custom extension builder that runs CMake.
    """

    def run(self) -> None:
        # Test for CMake
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as ex:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            ) from ex

        build_temp_path = Path(self.build_temp)
        build_directory = build_temp_path.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_directory}",
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,'$ORIGIN/../..'",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        # NOTE(brad): build_ext.editable_mode is not added until setuptools==64.0.0
        editable_mode = self.editable_mode if hasattr(self, "editable_mode") else False

        if editable_mode:
            # NOTE(brad): If in editable mode (i.e., building with pip install -e), we
            # place and expect cc_sym.so in the top-level of SOURCE_DIR. Since cc_sym needs
            # to be dynamically linked to its dependencies (libsymforce_gen.so
            # & libsymforce_opt.so) which it finds via the RPATH, we set the RPATH to
            # $ORIGIN (a keyword) which is just the directory of the lib at runtime (the macos
            # analog to $ORIGIN is @loader_path). We will later place the dependencies of
            # cc_sym.so in the same directory so that they will be found.
            if sys.platform == "linux" or sys.platform == "linux2":
                cmake_args.append("-DCMAKE_BUILD_RPATH=$ORIGIN")
            elif sys.platform == "darwin":
                cmake_args.append("-DCMAKE_BUILD_RPATH=@loader_path")

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        # NOTE(hayk): Don't build tests or examples through pip.
        cmake_args += [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DSYMFORCE_BUILD_TESTS=OFF",
            "-DSYMFORCE_BUILD_EXAMPLES=OFF",
        ]

        # Assuming Makefiles
        build_args += ["--", f"-j{multiprocessing.cpu_count()}"]

        self.build_args = build_args

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )

        build_temp_path.mkdir(parents=True, exist_ok=True)

        # CMakeLists.txt is in the same directory as this setup.py file
        print("-" * 10, "Running CMake prepare", "-" * 40)
        subprocess.run(
            ["cmake", str(SOURCE_DIR)] + cmake_args, cwd=self.build_temp, env=env, check=True
        )

        print("-" * 10, "Building extensions", "-" * 40)
        cmake_cmd = ["cmake", "--build", "."] + self.build_args
        subprocess.run(cmake_cmd, cwd=self.build_temp, check=True)

        if editable_mode:
            # NOTE(brad) When installed in editable mode, python expects to find symengine
            # in the symenginepy/symengine source directory. Everything is already there
            # except the compiled symengine_wrapper extension module, so we copy that there
            # as well.
            symengine_wrapper = maybe_find_symengine_wrapper(
                build_temp_path, self.get_ext_filename("symengine_wrapper")
            )
            # NOTE(brad) For some reason that I don't fully understand, the symengine_wrapper
            # shared library is neither present in symengine_install nor needed in the source
            # directory on macos sometimes (for example, on the github actions macos runner).
            # So, if the file's not present, we just move on.
            if symengine_wrapper:
                self.copy_file(
                    str(symengine_wrapper),
                    str(
                        SOURCE_DIR
                        / "third_party"
                        / "symenginepy"
                        / "symengine"
                        / "lib"
                        / self.get_ext_filename("symengine_wrapper")
                    ),
                )

            # NOTE(brad) By setting the RPATH of the generated binaries to include $ORIGIN,
            # we told all binaries they can expect to find any shared libraries in the same
            # directory they are already in. In particular, we told cc_sym.so it can find
            # its dependencies (the below shared libraries) in SOURCE_DIR (where we put it).
            # Thus, we need to copy the shared libaries there as well so that cc_sym.so will
            # find them when it looks.
            for cc_sym_dependency in build_temp_path.glob("libsymforce_*"):
                self.copy_file(
                    str(cc_sym_dependency),
                    str(SOURCE_DIR / cc_sym_dependency.name),
                )

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext: CMakeExtension) -> None:
        if ext.name == "lcmtypes":
            build_temp_path = Path(self.build_temp)
            dest_path_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
            if self.inplace:
                # NOTE(brad): If building in-place, place package in lcmtypes_build dir to avoid
                # collision with existing lcmtypes directory. Happens in editable installs.
                dest_path = dest_path_dir / "lcmtypes_build" / "lcmtypes"
            else:
                dest_path = dest_path_dir / "lcmtypes"

            self.copy_tree(
                str(build_temp_path / "lcmtypes" / "python2.7" / "lcmtypes"),
                str(dest_path),
            )
            return

        build_temp = Path(self.build_temp).resolve()
        extension_source_paths = {"cc_sym": build_temp / "pybind" / self.get_ext_filename("cc_sym")}

        build_temp = Path(self.build_temp).resolve()
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(extension_source_paths[ext.name], str(dest_path))


def maybe_rewrite_local_dependencies(dep_list: T.List[str]) -> T.List[str]:
    if "SYMFORCE_REWRITE_LOCAL_DEPENDENCIES" in os.environ:

        def filter_local(s: str) -> str:
            if "@" in s:
                s = f"{s.split('@')[0]}=={os.environ['SYMFORCE_REWRITE_LOCAL_DEPENDENCIES']}"
            return s

        return [filter_local(dependency) for dependency in dep_list]
    else:
        return dep_list


def maybe_find_symengine_wrapper(build_dir: Path, ext_filename: str) -> T.Optional[Path]:
    symengine_wrapper_candidates = list(
        build_dir.glob(
            f"symengine_install/**/lib/python{sys.version_info.major}.{sys.version_info.minor}/*-packages/symengine/lib/{ext_filename}"
        )
    )

    if len(symengine_wrapper_candidates) > 1:
        raise FileNotFoundError(
            f"Expected to find exactly one symengine_wrapper.so, but found {len(symengine_wrapper_candidates)}: {symengine_wrapper_candidates}"
        )

    return next(iter(symengine_wrapper_candidates), None)


def find_symengine_wrapper(build_dir: Path, ext_filename: str) -> Path:
    symengine_wrapper = maybe_find_symengine_wrapper(build_dir, ext_filename)
    if symengine_wrapper is None:
        raise FileNotFoundError(f"Could not find symengine_wrapper.so in {build_dir}")
    return symengine_wrapper


class InstallWithExtras(install):
    """
    Custom install step that:
        1) Installs symenginepy so it can be imported
        2) Installs additional shared libraries needed by cc_sym (e.g. libmetis)
        3) Installs lcmtypes python package
    """

    def run(self) -> None:
        super().run()

        build_ext_obj = self.distribution.get_command_obj("build_ext")
        assert isinstance(build_ext_obj, CMakeBuild)
        build_dir = Path(build_ext_obj.build_temp)

        # Install symengine
        # NOTE(aaron): We add symenginepy as a package down below, and the only remaining thing we
        # need is the compiled symengine_wrapper.so, which we move into place here.  This doesn't
        # include the additional Cython sources that symenginepy includes in their distributions,
        # but I'm honestly not sure why they include them or why you'd need them.
        self.copy_file(
            str(
                find_symengine_wrapper(
                    build_dir, build_ext_obj.get_ext_filename("symengine_wrapper")
                )
            ),
            Path.cwd()
            / self.install_platlib
            / "symengine"
            / "lib"
            / build_ext_obj.get_ext_filename("symengine_wrapper"),
        )

        # Configure with install prefix
        subprocess.run(
            ["cmake", str(SOURCE_DIR), f"-DCMAKE_INSTALL_PREFIX={self.prefix}"],
            cwd=build_dir,
            check=True,
        )

        # Install other libraries needed for cc_sym.so
        subprocess.run(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=build_dir,
            check=True,
        )


setup_requirements = [
    "setuptools>=62.3.0",  # For package data globs
    "setuptools-scm>=8",
    "wheel",
    "pip",
    "cmake>=3.17",
    "cython>=0.19.1,<3",
    f"skymarshal @ file://localhost/{ESCAPED_SOURCE_DIR}/third_party/skymarshal",
]

docs_requirements = [
    "furo",
    "ipykernel",
    # nbconvert depends on this, but doesn't specify the dependency
    "ipython-genutils",
    "matplotlib",
    "myst-parser",
    "nbsphinx",
    "nbstripout",
    "pandas",
    "plotly",
    "Sphinx",
    "sphinx-copybutton",
    "sphinxext-opengraph",
    "breathe",
]


def symforce_rev() -> str:
    """
    Get the current git sha, falling back to `main`.  Only used for images in the README, where it
    isn't super important that this is very robust
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], check=True, text=True, stdout=subprocess.PIPE
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "main"


def fixed_readme() -> str:
    """
    Fix things in the README for PyPI
    """
    readme = Path("README.md").read_text(encoding="UTF8")

    # Replace relative links with absolute, so images appear correctly on PyPI
    readme = readme.replace(
        "docs/static/images/",
        f"https://raw.githubusercontent.com/symforce-org/symforce/{symforce_rev()}/docs/static/images/",
    )

    # Remove the DARK_MODE_ONLY tags
    # https://github.com/pypi/warehouse/issues/11251
    # See https://stackoverflow.com/a/1732454/2791611
    readme = re.sub(
        r"<!--\s*DARK_MODE_ONLY\s*-->((?!DARK_MODE_ONLY).)*<!--\s*/DARK_MODE_ONLY\s*-->",
        "",
        readme,
        flags=re.MULTILINE | re.DOTALL,
    )

    return readme


if __name__ == "__main__":
    setup(
        long_description=fixed_readme(),
        long_description_content_type="text/markdown",
        # The SymForce package is a namespace package (important for data-only subdirectories
        # specifically).  We could also treat the others as namespace packages, but that makes it
        # more annoying to exclude non-package directories.
        packages=find_namespace_packages(where=".", include=["symforce*"])
        + find_packages(where=".", exclude=["symforce*"])
        + find_packages(where="third_party/symenginepy"),
        package_dir={
            "symforce": "symforce",
            "symengine": "third_party/symenginepy/symengine",
            "lcmtypes": "lcmtypes_build/lcmtypes",
        },
        package_data={
            "": [
                "*.jinja",
                "*.mtx",
                "README*",
                ".clang-format",
                "py.typed",
                "ruff.toml",
                "prettier.config.mjs",
            ]
        },
        # pyproject.toml doesn't allow specifying url or homepage separately, and if it's not
        # specified separately PyPI sorts all the links alphabetically
        # https://github.com/pypi/warehouse/issues/3097
        url="https://symforce.org",
        # Override the extension builder with our cmake class
        cmdclass=dict(
            build_ext=CMakeBuild,
            install=InstallWithExtras,
            develop=PatchDevelop,
        ),
        # Build C++ extension module
        ext_modules=[CMakeExtension("cc_sym"), CMakeExtension("lcmtypes")],
        # Barebones packages needed to run symforce
        install_requires=maybe_rewrite_local_dependencies(
            [
                "ruff",
                # clang-format 21.1.3 and newer have this issue:
                # https://github.com/llvm/llvm-project/issues/170573
                "clang-format<21.1.3",
                "graphviz",
                "jinja2",
                "numpy",
                "scipy",
                f"skymarshal @ file://localhost/{ESCAPED_SOURCE_DIR}/third_party/skymarshal",
                "sympy>=1.11",
                f"symforce-sym @ file://localhost/{ESCAPED_SOURCE_DIR}/gen/python",
                "typing-extensions; python_version<'3.9'",
                "sortedcontainers",
            ]
        ),
        setup_requires=setup_requirements,
        extras_require={
            "docs": maybe_rewrite_local_dependencies(docs_requirements),
            "dev": maybe_rewrite_local_dependencies(
                docs_requirements
                + [
                    "argh",
                    "coverage",
                    "jinja2~=3.0",
                    "mypy~=1.14.0",
                    "numba",
                    # Base for https://github.com/sizmailov/pybind11-stubgen/pull/263
                    "pybind11-stubgen>=2.5.5",
                    "ruff~=0.9.7",
                    "types-jinja2",
                    "types-requests",
                    "types-setuptools",
                    "sortedcontainers-stubs",
                    # Oldest version that solves to the right requirements
                    "uv>=0.2.0",
                ]
            ),
            "setup": maybe_rewrite_local_dependencies(setup_requirements),
        },
    )
