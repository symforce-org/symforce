# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from lcmtypes.sym._lambda_update_type_t import lambda_update_type_t
from lcmtypes.sym._optimizer_params_t import optimizer_params_t


@dataclass
class OptimizerParams:
    """
    Parameters for the SymForce Optimizer

    Mirrors the ``optimizer_params_t`` LCM type, see documentation there for information on each
    parameter.
    """

    verbose: bool = False
    debug_stats: bool = False
    check_derivatives: bool = False
    include_jacobians: bool = False
    debug_checks: bool = False
    initial_lambda: float = 1.0
    lambda_lower_bound: float = 0.0
    lambda_upper_bound: float = 1000000.0
    lambda_update_type: lambda_update_type_t = lambda_update_type_t.STATIC
    lambda_up_factor: float = 4.0
    lambda_down_factor: float = 1 / 4.0
    dynamic_lambda_update_beta: float = 2.0
    dynamic_lambda_update_gamma: float = 3.0
    dynamic_lambda_update_p: int = 3
    use_diagonal_damping: bool = False
    use_unit_damping: bool = True
    keep_max_diagonal_damping: bool = False
    diagonal_damping_min: float = 1e-6
    iterations: int = 50
    early_exit_min_reduction: float = 1e-6
    enable_bold_updates: bool = False

    def to_lcm(self) -> optimizer_params_t:
        return optimizer_params_t(**dataclasses.asdict(self))

    @staticmethod
    def from_lcm(msg: optimizer_params_t) -> OptimizerParams:
        return OptimizerParams(**{k: getattr(msg, k) for k in msg.__slots__})
