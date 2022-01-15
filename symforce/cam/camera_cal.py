from symforce import geo
from symforce.ops.interfaces import Storage
from symforce import sympy as sm
from symforce import typing as T


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
        self.distortion_coeffs = geo.M(distortion_coeffs)
        self.focal_length = geo.V2(focal_length)
        self.principal_point = geo.V2(principal_point)

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
        return cls(focal_length=vec[0:2], principal_point=vec[2:4], distortion_coeffs=vec[4:])

    @classmethod
    def symbolic(cls: T.Type[CameraCalT], name: str, **kwargs: T.Any) -> CameraCalT:
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
        self, point: geo.V3, epsilon: T.Scalar = 0
    ) -> T.Tuple[geo.V2, T.Scalar]:
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Return:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds else 0
        """
        raise NotImplementedError()

    def camera_ray_from_pixel(
        self, pixel: geo.V2, epsilon: T.Scalar = 0
    ) -> T.Tuple[geo.V3, T.Scalar]:
        """
        Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

        TODO(hayk): Add a normalize boolean argument? Like in `cam.Camera`

        Return:
            camera_ray: The ray in the camera frame (NOT normalized)
            is_valid: 1 if the operation is within bounds else 0
        """
        raise NotImplementedError()
