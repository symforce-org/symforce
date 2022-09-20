# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce.internal.symbolic as sf
from symforce import geo
from symforce import ops
from symforce import type_helpers
from symforce import typing as T
from symforce.ops.interfaces import Storage


class CameraCal(Storage):
    """
    Base class for symbolic camera models.
    """

    # Type that represents this or any subclasses
    CameraCalT = T.TypeVar("CameraCalT", bound="CameraCal")

    NUM_DISTORTION_COEFFS = 0

    def __init__(
        self,
        focal_length: T.Sequence[T.Scalar],
        principal_point: T.Sequence[T.Scalar],
        distortion_coeffs: T.Sequence[T.Scalar] = tuple(),
    ) -> None:
        assert len(distortion_coeffs) == self.NUM_DISTORTION_COEFFS
        self.focal_length = geo.V2(focal_length)
        self.principal_point = geo.V2(principal_point)
        self.distortion_coeffs = geo.M(distortion_coeffs)

    @classmethod
    def from_distortion_coeffs(
        cls: T.Type[CameraCalT],
        focal_length: T.Sequence[T.Scalar],
        principal_point: T.Sequence[T.Scalar],
        distortion_coeffs: T.Sequence[T.Scalar] = tuple(),
    ) -> CameraCalT:
        """
        Construct a Camera Cal of type cls from the focal_length, principal_point, and distortion_coeffs.

        Note, some subclasses may not allow symbolic arguments unless additional keyword arguments are passed in.
        """
        instance = cls.__new__(cls)
        instance.focal_length = geo.V2(focal_length)
        instance.principal_point = geo.V2(principal_point)
        instance.distortion_coeffs = geo.M(distortion_coeffs)
        return instance

    @classmethod
    def storage_order(cls) -> T.Tuple[T.Tuple[str, int], ...]:
        """
        Return list of the names of values returned in the storage paired with
        the dimension of each value.
        """
        return ("focal_length", 2), ("principal_point", 2)

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    @classmethod
    def storage_dim(cls) -> int:
        return 4 + cls.NUM_DISTORTION_COEFFS

    def to_storage(self) -> T.List[T.Scalar]:
        return (
            self.focal_length.to_storage()
            + self.principal_point.to_storage()
            + self.distortion_coeffs.to_storage()
        )

    @classmethod
    def from_storage(cls: T.Type[CameraCalT], vec: T.Sequence[T.Scalar]) -> CameraCalT:
        assert len(vec) == cls.storage_dim()
        return cls.from_distortion_coeffs(
            focal_length=vec[0:2], principal_point=vec[2:4], distortion_coeffs=vec[4:]
        )

    @classmethod
    def symbolic(cls: T.Type[CameraCalT], name: str, **kwargs: T.Any) -> CameraCalT:
        with sf.scope(name):
            if cls.NUM_DISTORTION_COEFFS > 0:
                return cls.from_distortion_coeffs(
                    focal_length=sf.symbols("f_x f_y"),
                    principal_point=sf.symbols("c_x c_y"),
                    distortion_coeffs=geo.Matrix(cls.NUM_DISTORTION_COEFFS, 1)
                    .symbolic("C", **kwargs)
                    .to_flat_list(),
                )
            else:
                return cls(
                    focal_length=sf.symbols("f_x f_y"), principal_point=sf.symbols("c_x c_y")
                )

    def __repr__(self) -> str:
        return "<{}\n  focal_length={},\n  principal_point={},\n  distortion_coeffs={}>".format(
            self.__class__.__name__,
            self.focal_length.to_storage(),
            self.principal_point.to_storage(),
            self.distortion_coeffs.to_storage(),
        )

    def parameters(self) -> T.List[T.Scalar]:
        return (
            self.focal_length.to_storage()
            + self.principal_point.to_storage()
            + self.distortion_coeffs.to_storage()
        )

    # -------------------------------------------------------------------------
    # Required camera methods
    # -------------------------------------------------------------------------

    def pixel_from_camera_point(
        self, point: geo.V3, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V2, T.Scalar]:
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Return:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds else 0
        """
        raise NotImplementedError()

    def camera_ray_from_pixel(
        self, pixel: geo.V2, epsilon: T.Scalar = sf.epsilon()
    ) -> T.Tuple[geo.V3, T.Scalar]:
        """
        Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

        TODO(hayk): Add a normalize boolean argument? Like in `cam.Camera`

        Return:
            camera_ray: The ray in the camera frame (NOT normalized)
            is_valid: 1 if the operation is within bounds else 0
        """
        raise NotImplementedError()

    @classmethod
    def has_camera_ray_from_pixel(cls) -> bool:
        """
        Returns True if cls has implemented the method camera_ray_from_pixel, and False
        otherwise.
        """
        try:
            type_helpers.symbolic_eval(cls.camera_ray_from_pixel)
        except NotImplementedError:
            return False
        return True


# Register ops
from symforce.ops.impl.vector_class_lie_group_ops import VectorClassLieGroupOps

ops.LieGroupOps.register(CameraCal, VectorClassLieGroupOps)
