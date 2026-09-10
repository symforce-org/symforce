# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce

symforce.set_epsilon_to_symbol()
symforce.set_log_level("info")

# In general, you want to make sure symengine is installed and being used correctly, the sympy
# symbolic API is very slow
assert symforce.get_symbolic_api() == "symengine"

import plotly.graph_objects as go
from tqdm import tqdm

import symforce.symbolic as sf
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.opt.optimizer import OptimizerParams
from symforce.values import Values

from . import pgo_dataset


def main() -> None:
    ids, edges, values = pgo_dataset.load("parking-garage.g2o")

    values["epsilon"] = sf.numeric_epsilon

    print(f"Num poses: {len(ids)}")
    print(f"Num edges: {len(edges)}")

    def residual(
        world_T_p0: sf.Pose3,
        world_T_p1: sf.Pose3,
        p0_T_p1: sf.Pose3,
        sqrt_info: sf.M66,
        epsilon: sf.Scalar,
    ) -> sf.M:
        unwhitened_residual = sf.V6(
            (world_T_p0.inverse() * world_T_p1 * p0_T_p1.inverse()).to_tangent(epsilon)
        )
        return sqrt_info * unwhitened_residual

    factors = []
    for i, j in tqdm(edges):
        factors.append(
            Factor([f"p_{i}", f"p_{j}", f"e_{i}_{j}", f"i_{i}_{j}", "epsilon"], residual)
        )

    params = OptimizerParams(debug_stats=True, verbose=True)

    optimizer = Optimizer(factors, [f"p_{i}" for i in ids], params=params)

    result = optimizer.optimize(values)

    def draw_graph(values: Values, name: str) -> tuple[go.Scatter3d, go.Scatter3d]:
        poses = {i: values[f"p_{i}"] for i in ids}
        trace1 = go.Scatter3d(
            x=[poses[i].t[0, 0] for i in ids],
            y=[poses[i].t[1, 0] for i in ids],
            z=[poses[i].t[2, 0] for i in ids],
            mode="markers",
            name=f"{name} Points",
        )
        x_lines, y_lines, z_lines = [], [], []
        for e in edges:
            for i in range(2):
                x_lines.append(poses[e[i]].t[0, 0])
                y_lines.append(poses[e[i]].t[1, 0])
                z_lines.append(poses[e[i]].t[2, 0])
            x_lines.append(None)
            y_lines.append(None)
            z_lines.append(None)

        trace2 = go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode="lines", name=f"{name} Edges")

        return trace1, trace2

    fig = go.Figure(
        data=draw_graph(values, "Original") + draw_graph(result.optimized_values, "Optimized")
    )
    fig.show()


if __name__ == "__main__":
    main()
