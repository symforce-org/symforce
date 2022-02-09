from .class_storage_ops import ClassStorageOps
from .abstract_vector_lie_group_ops import AbstractVectorLieGroupOps


class VectorClassLieGroupOps(ClassStorageOps, AbstractVectorLieGroupOps):
    """
    A generic implementation of Lie group ops for subclasses of symforce.ops.interfaces.Storage.

    Treats the subclass like R^n where the vector is the storage representation.

    To elaborate, treats the subclass as a Lie group whose identity is the zero vector, group
    operation is vector addition, and whose vector representation is given by the to_storage
    operation.
    """
