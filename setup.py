# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import distutils.errors
import distutils.util
import multiprocessing
import os
import re
import subprocess
import sys
import typing as T
from pathlib import Path

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

SOURCE_DIR = Path(__file__).resolve().parent


def symforce_version() -> str:
    """
    Fetch the current symforce version from _version.py

    We can't import the version here, so we have to do some text parsing
    """
    version_file_contents = (Path(__file__).parent / "symforce" / "_version.py").read_text()
    version_match = re.search(r'^version = "(.+)"$', version_file_contents, flags=re.MULTILINE)
    assert version_match is not None
    return version_match.group(1)


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

    def run(self) -> None:  # type: ignore[override]
        # NOTE(brad): In setuptools 64.0.0, the editable_mode field of the
        # build_ext was added and is set to True during a pep 660 editable
        # install.
        # Since we want to use this field in our CMakeBuild extension of
        # build_ext to identify if we're performing an editable install
        # regardless of whether we're building with develop or build_editable,
        # (both before and after version 64.0.0), we patch develop to add this
        # field to build_ext and set it to True.
        self.distribution.get_command_obj("build_ext").editable_mode = True  # type: ignore[attr-defined]
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

        self.build_args = build_args  # pylint: disable=attribute-defined-outside-init

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
            symengine_wrapper = next(
                build_temp_path.glob(
                    f"symengine_install/**/lib/python{sys.version_info.major}.{sys.version_info.minor}/*-packages/symengine/lib/{self.get_ext_filename('symengine_wrapper')}"
                ),
                None,
            )
            # NOTE(brad) For some reason that I don't fully understand, the symengine_wrapper
            # shared library is neither present in symengine_install nor needed in the source
            # directory on macos sometimes (for example, on the github actions macos runner).
            # So, if the file's not present, we just move on.
            if symengine_wrapper:
                self.copy_file(
                    symengine_wrapper,
                    SOURCE_DIR
                    / "third_party"
                    / "symenginepy"
                    / "symengine"
                    / "lib"
                    / self.get_ext_filename("symengine_wrapper"),
                )

            # NOTE(brad) By setting the RPATH of the generated binaries to include $ORIGIN,
            # we told all binaries they can expect to find any shared libraries in the same
            # directory they are already in. In particular, we told cc_sym.so it can find
            # its dependencies (the below shared libraries) in SOURCE_DIR (where we put it).
            # Thus, we need to copy the shared libaries there as well so that cc_sym.so will
            # find them when it looks.
            for cc_sym_dependency in build_temp_path.glob("libsymforce_*"):
                self.copy_file(
                    cc_sym_dependency,
                    SOURCE_DIR / cc_sym_dependency.name,
                )

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext: CMakeExtension) -> None:
        if ext.name == "lcmtypes":  # type: ignore[attr-defined]
            build_temp_path = Path(self.build_temp)
            dest_path_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent  # type: ignore[attr-defined]
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
        dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()  # type: ignore[attr-defined]
        dest_directory = dest_path.parents[0]
        dest_directory.mkdir(parents=True, exist_ok=True)
        self.copy_file(extension_source_paths[ext.name], dest_path)  # type: ignore[attr-defined]


class SymForceEggInfo(egg_info):
    """
    Custom Egg info, that optionally rewrites local dependencies (that are stored in the symforce
    repo) to package dependencies
    """

    user_options = egg_info.user_options + [
        (
            "rewrite-local-dependencies=",
            None,
            "Rewrite in-repo Python dependencies from `package @ file:` to `package==version`.  Can"
            " also be provided as the environment variable SYMFORCE_REWRITE_LOCAL_DEPENDENCIES",
        )
    ]

    def initialize_options(self) -> None:
        super().initialize_options()
        self.rewrite_local_dependencies = False  # pylint: disable=attribute-defined-outside-init

    def finalize_options(self) -> None:
        super().finalize_options()

        if not isinstance(self.rewrite_local_dependencies, bool):
            self.rewrite_local_dependencies = (  # pylint: disable=attribute-defined-outside-init
                bool(distutils.util.strtobool(self.rewrite_local_dependencies))
            )

        if "SYMFORCE_REWRITE_LOCAL_DEPENDENCIES" in os.environ:
            self.rewrite_local_dependencies = (  # pylint: disable=attribute-defined-outside-init
                bool(distutils.util.strtobool(os.environ["SYMFORCE_REWRITE_LOCAL_DEPENDENCIES"]))
            )

    def run(self) -> None:
        # Rewrite dependencies from the local `file:` versions to generic pinned package versions.
        # This is what we want when e.g. building wheels for PyPI, where those local dependencies
        # are hosted as their own PyPI packages.  We can't just decide whether to do this e.g. based
        # on whether we're building a wheel, since `pip install .` also builds a wheel to install.
        if self.rewrite_local_dependencies:

            def filter_local(s: str) -> str:
                if "@" in s:
                    s = f"{s.split('@')[0]}=={symforce_version()}"
                return s

            self.distribution.install_requires = [  # type: ignore[attr-defined]
                filter_local(requirement) for requirement in self.distribution.install_requires  # type: ignore[attr-defined]
            ]
            self.distribution.extras_require = {  # type: ignore[attr-defined]
                k: [filter_local(requirement) for requirement in v]
                for k, v in self.distribution.extras_require.items()  # type: ignore[attr-defined]
            }

        super().run()


class InstallWithExtras(install):
    """
    Custom install step that:
        1) Installs symenginepy so it can be imported
        2) Installs additional shared libraries needed by cc_sym (e.g. libmetis)
        3) Installs lcmtypes python package
    """

    def run(self) -> None:
        super().run()

        build_ext_obj = self.distribution.get_command_obj("build_ext")  # type: ignore[attr-defined]
        build_dir = Path(self.distribution.get_command_obj("build_ext").build_temp)  # type: ignore[attr-defined]

        # Install symengine
        # NOTE(aaron): We add symenginepy as a package down below, and the only remaining thing we
        # need is the compiled symengine_wrapper.so, which we move into place here.  This doesn't
        # include the additional Cython sources that symenginepy includes in their distributions,
        # but I'm honestly not sure why they include them or why you'd need them.
        self.copy_file(
            build_dir
            / "symengine_install"
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "symengine"
            / "lib"
            / build_ext_obj.get_ext_filename("symengine_wrapper"),
            Path.cwd()
            / self.install_platlib  # type: ignore[attr-defined]
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
    "setuptools",
    "wheel",
    "pip",
    "cmake>=3.17",
    "cython>=0.19.1",
    f"skymarshal @ file://localhost/{SOURCE_DIR}/third_party/skymarshal",
]

docs_requirements = [
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
    # sphinx-autodoc-typehints >=1.15 contains a bug causing it to crash parsing our typing.py
    "sphinx-autodoc-typehints<1.15",
    "breathe",
]

cmdclass: T.Dict[str, T.Any] = dict(
    build_ext=CMakeBuild,
    install=InstallWithExtras,
    egg_info=SymForceEggInfo,
    develop=PatchDevelop,
)


def symforce_data_files() -> T.List[str]:
    # package_data doesn't support recursive globs until this merges:
    # https://github.com/pypa/setuptools/pull/3309
    # So, we do the globbing ourselves
    SYMFORCE_PKG_DIR = SOURCE_DIR / "symforce"
    files_with_pattern = lambda pattern: [
        str(p.relative_to(SYMFORCE_PKG_DIR)) for p in SYMFORCE_PKG_DIR.rglob(pattern)
    ]
    return (
        files_with_pattern("*.jinja")
        + files_with_pattern("*.mtx")
        + ["test_util/random_expressions/README", ".clang-format"]
    )


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
    readme = Path("README.md").read_text()

    # Replace relative links with absolute, so images appear correctly on PyPI
    readme = readme.replace(
        "docs/static/images/",
        f"https://raw.githubusercontent.com/symforce-org/symforce/{symforce_rev()}/docs/static/images/",
    )

    # Remove the DARK_MODE_ONLY tags
    # See https://stackoverflow.com/a/1732454/2791611
    readme = re.sub(
        r"<!--\s*DARK_MODE_ONLY\s*-->((?!DARK_MODE_ONLY).)*<!--\s*/DARK_MODE_ONLY\s*-->",
        "",
        readme,
        flags=re.MULTILINE | re.DOTALL,
    )

    return readme


setup(
    name="symforce",
    version=symforce_version(),
    author="Skydio, Inc",
    author_email="hayk@skydio.com",
    description="Fast symbolic computation, code generation, and nonlinear optimization for robotics",
    keywords="python computer-vision cpp robotics optimization structure-from-motion motion-planning code-generation slam autonomous-vehicles symbolic-computation",
    license="Apache 2.0",
    license_file="LICENSE",
    long_description=fixed_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/symforce-org/symforce",
    project_urls={
        "Bug Tracker": "https://github.com/symforce-org/symforce/issues",
        "Source": "https://github.com/symforce-org/symforce",
    },
    # For a list of valid classifiers see https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Embedded Systems",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education :: Computer Aided Instruction (CAI)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: C++",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    # -------------------------------------------------------------------------
    # Build info
    # -------------------------------------------------------------------------
    # Minimum Python version
    python_requires=">=3.8",
    # Find all packages in the directory
    packages=find_packages() + find_packages(where="third_party/symenginepy"),
    package_data={
        "symforce": symforce_data_files(),
    },
    package_dir={
        "symforce": "symforce",
        "symengine": "third_party/symenginepy/symengine",
        "lcmtypes": "lcmtypes_build/lcmtypes",
    },
    # Override the extension builder with our cmake class
    cmdclass=cmdclass,
    # Build C++ extension module
    ext_modules=[CMakeExtension("cc_sym"), CMakeExtension("lcmtypes")],
    # Barebones packages needed to run symforce
    install_requires=[
        "black",
        "clang-format",
        "graphviz",
        "jinja2",
        "numpy",
        "scipy",
        f"skymarshal @ file://localhost/{SOURCE_DIR}/third_party/skymarshal",
        "sympy~=1.11.1",
        f"symforce-sym @ file://localhost/{SOURCE_DIR}/gen/python",
    ],
    setup_requires=setup_requirements,
    extras_require={
        "docs": docs_requirements,
        "dev": docs_requirements
        + [
            "click~=8.0.4",  # Required by black, but not specified by our version of black
            "argh",
            "black[jupyter]==21.12b0",
            "coverage",
            "isort",
            "jinja2~=3.0.3",
            "mypy==0.910",
            "numba",
            "pip-tools",
            "pybind11-stubgen",
            "pylint",
            "types-jinja2",
            "types-pygments",
            "types-requests",
            "types-setuptools",
        ],
        "_setup": setup_requirements,
    },
    # Not okay to zip
    zip_safe=False,
)
