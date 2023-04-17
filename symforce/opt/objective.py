# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from abc import ABC
from abc import abstractmethod

from symforce import typing as T
from symforce.opt.residual_block import ResidualBlock
from symforce.values import Values


class Objective(ABC):
    """
    An objective, defined as a residual or group of residuals and an associated Params block

    Subclasses should add individual residual functions as static methods that return
    `ResidualBlock`, and define the Params block as a dataclass
    """

    Params: T.Dataclass

    @staticmethod
    def default_inputs(enabled: bool) -> Values:
        """
        Should return instantiated numerical arguments for the residual function(s) of this
        objective, to be used for testing and sanity checking

        Args:
            enabled: Whether to configure the objective params with the objective enabled or not
                     (e.g. setting costs to 0)
        """
        raise NotImplementedError()


class TimestepObjective(Objective):
    """
    An objective defined as a single residual applied over multiple timesteps, and associated Params
    block

    Subclasses should define the residual_at_timestep function needed to compute the residual at
    each timestep.
    """

    @T.any_args
    @abstractmethod
    def residual_at_timestep(self, *args: T.Any) -> ResidualBlock:
        """
        Compute the residual at a single timestep

        Args:
            *: Any arguments needed for the particular residual; typically these are expressions
                evaluated at a particular time, or single element of a timestepped sequence

        Returns:
            residual_block: The ResidualBlock for this objective at this timestep
        """
        pass
