# aclint: py2 py3
"Allow the skymarshal package to be executable"
from __future__ import absolute_import

from skymarshal import skymarshal
from skymarshal.emit_cpp import SkymarshalCpp
from skymarshal.emit_djinni import SkymarshalDjinni
from skymarshal.emit_java import SkymarshalJava
from skymarshal.emit_proto import SkymarshalProto, SkymarshalProtoLCM
from skymarshal.emit_python import SkymarshalPython
from skymarshal.emit_typescript import SkymarshalTypeScript

if __name__ == "__main__":
    skymarshal.main(
        [
            SkymarshalCpp,
            SkymarshalDjinni,
            SkymarshalProto,
            SkymarshalProtoLCM,
            SkymarshalJava,
            SkymarshalPython,
            SkymarshalTypeScript,
        ]
    )
