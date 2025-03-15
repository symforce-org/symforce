# ----------------------------------------------------------------------------
# SymForce - Copyright 2025, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np

from symforce.experimental.caspar.examples.bal_deployed.generated import (  # type: ignore[import-not-found]
    caspar_lib,
)

try:
    npz_path = Path(sys.argv[1])
    assert npz_path.exists() and npz_path.suffix == ".npz"
except (IndexError, AssertionError):
    print("Error: sys.argv[1] should be /path/to/data.npz")
    sys.exit(1)

cam_ids, point_ids, camdata, pointdata, pixels = np.load(npz_path, allow_pickle=False).values()

params = caspar_lib.SolverParams()
params.diag_init = 1.0
params.solver_iter_max = 100
params.pcg_iter_max = 10
params.pcg_rel_error_exit = 1e-2

solver = caspar_lib.GraphSolver(
    params,
    Cam_num_max=camdata.shape[0],
    Point_num_max=pointdata.shape[0],
    fac_reprojection_num_max=pixels.shape[0],
)
solver.set_Cam_nodes_from_stacked_host(camdata, 0)
solver.set_Point_nodes_from_stacked_host(pointdata, 0)
solver.set_fac_reprojection_pixel_data_from_stacked_host(pixels, 0)
solver.set_fac_reprojection_cam_indices_from_host(cam_ids)
solver.set_fac_reprojection_point_indices_from_host(point_ids)
solver.finish_indices()
print("starting")
solver.solve(print_progress=True)
