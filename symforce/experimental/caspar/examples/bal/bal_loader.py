# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import typing as T
from pathlib import Path
from urllib import request

import numpy as np

import symforce.symbolic as sf


def load_bal(
    name: str = "venice/problem-1778-993923-pre.txt.bz2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fpath = Path(__file__).resolve().parent / f"data/{name}"
    fpath.parent.mkdir(exist_ok=True, parents=True)

    if not fpath.exists():
        print("Downloading data... (this may take a while the first time)")
        pre = "https://grail.cs.washington.edu/projects/bal"
        request.urlretrieve(f"{pre}/{'/'.join(fpath.parts[-3:])}", fpath)

    if not (npz_path := fpath.with_suffix(".npz")).exists():
        print("Loading data... (this may take a while the first time)")

        def load(
            file: Path,
            typ: T.Type[np.number[T.Any]],
            start: int,
            nrows: int,
            cols: int | T.Sequence[int] | None = None,
        ) -> np.ndarray:
            a = np.loadtxt(file, typ, skiprows=start, max_rows=nrows, usecols=cols)  # type: ignore[arg-type]
            return a

        n_cams, n_points, n_facs = load(fpath, np.int32, 0, 1)
        cam_ids, point_ids = np.ascontiguousarray(load(fpath, np.int32, 1, n_facs, (0, 1)).T)
        pixels = load(fpath, np.float32, 1, n_facs, (2, 3))
        camdata_tangent = load(fpath, np.float32, 1 + n_facs, n_cams * 9)
        camdata_tangent = camdata_tangent.reshape(n_cams, 9)
        camdata = np.array(
            [
                [*sf.Pose3(sf.Rot3.from_tangent(d[:3]), sf.V3(d[3:6])).to_storage(), *d[6:]]
                for d in camdata_tangent.astype(float)
            ],
            dtype=np.float32,
        )
        points = load(fpath, np.float32, 1 + n_facs + n_cams * 9, n_points * 3)
        points = points.reshape(n_points, 3)

        np.savez_compressed(npz_path, cam_ids, point_ids, camdata, points, pixels)
    else:
        data = np.load(npz_path, allow_pickle=False)
        cam_ids, point_ids, camdata, points, pixels = data.values()
    return cam_ids, point_ids, camdata, points, pixels
