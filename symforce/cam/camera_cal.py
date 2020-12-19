from symforce import geo
from symforce import ops
from symforce.ops.interfaces import Storage
from symforce import sympy as sm
from symforce import types as T


class CameraCal(Storage):
    """
    Base class for symbolic camera models.
    """

    # Type that represents this or any subclasses
    CameraCalT = T.TypeVar("CameraCalT", bound="CameraCal")

    NUM_DISTORTION_COEFFS = 0

    def __init__(self, focal_length, principal_point, distortion_coeffs=tuple()):
        # type: (T.Sequence[T.Scalar], T.Sequence[T.Scalar], T.Sequence[T.Scalar]) -> None
        assert len(distortion_coeffs) == self.NUM_DISTORTION_COEFFS
        self.distortion_coeffs = geo.M(distortion_coeffs)
        self.focal_length = geo.V2(focal_length)
        self.principal_point = geo.V2(principal_point)

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    @classmethod
    def storage_dim(cls):
        # type: () -> int
        return 4 + cls.NUM_DISTORTION_COEFFS

    def to_storage(self):
        # type: () -> T.List[T.Scalar]
        return (
            self.focal_length.to_storage()
            + self.principal_point.to_storage()
            + self.distortion_coeffs.to_storage()
        )

    @classmethod
    def from_storage(cls, vec):
        # type: (T.Type[CameraCalT], T.Sequence[T.Scalar]) -> CameraCalT
        assert len(vec) == cls.storage_dim()
        return cls(focal_length=vec[0:2], principal_point=vec[2:4], distortion_coeffs=vec[4:])

    @classmethod
    def symbolic(cls, name, **kwargs):
        # type: (T.Type[CameraCalT], str, T.Any) -> CameraCalT
        with sm.scope(name):
            if cls.NUM_DISTORTION_COEFFS > 0:
                return cls(
                    focal_length=sm.symbols("f_x f_y"),
                    principal_point=sm.symbols("c_x c_y"),
                    distortion_coeffs=geo.Matrix(cls.NUM_DISTORTION_COEFFS, 1)
                    .symbolic("C", **kwargs)
                    .to_flat_list(),
                )
            else:
                return cls(
                    focal_length=sm.symbols("f_x f_y"), principal_point=sm.symbols("c_x c_y"),
                )

    def __repr__(self):
        # type: () -> str
        return "<{}\n  focal_length={},\n  principal_point={},\n  distortion_coeffs={}>".format(
            self.__class__.__name__,
            self.focal_length.to_storage(),
            self.principal_point.to_storage(),
            self.distortion_coeffs.to_storage(),
        )

    # -------------------------------------------------------------------------
    # Required camera methods
    # -------------------------------------------------------------------------

    def pixel_from_camera_point(self, point, epsilon=0):
        # type: (geo.Matrix31, T.Scalar) -> T.Tuple[geo.Matrix21, T.Scalar]
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Return:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds else 0
        """
        raise NotImplementedError()

    def camera_ray_from_pixel(self, pixel, epsilon=0):
        # type: (geo.Matrix21, T.Scalar) -> T.Tuple[geo.Matrix31, T.Scalar]
        """
        Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

        Return:
            camera_ray: The ray in the camera frame (NOT normalized)
            is_valid: 1 if the operation is within bounds else 0
        """
        raise NotImplementedError()
