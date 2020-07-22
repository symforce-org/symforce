import collections
import contextlib
import numpy as np

from symforce import logger
from symforce import sympy as sm
from symforce import types as T
from symforce import geo
from symforce import cam
from symforce import initialization
from symforce import ops
from symforce.ops.interfaces import Storage

from .attr_accessor import AttrAccessor


class Values(object):
    """
    Ordered dictionary serializable storage. This class is the basis for specifying both inputs
    and outputs in symforce. The hierarchy of nested values get code generated into types, and
    several methods of access and introspection are provided that reduce downstream complexity.

    Includes standard operator[] access to keys and values.

    Attributes:
        attr: Access with dot notation, such as `v.attr.states.x0` instead of `v['states.x0']`.
    """

    def __init__(self, **kwargs):
        # type: (T.Any) -> None
        """
        Create like a Python dict.

        Args:
            kwargs (dict): Initial values
        """
        # Underlying storage - ordered dictionary
        self.dict = collections.OrderedDict()  # type: T.Dict[str, T.Any]

        # Allow dot notation through this member
        # ex: v.attr.foo.bar = 12
        self.attr = AttrAccessor(self.dict)

        # Create context manager helpers for .scope()
        self.__scopes__ = []  # type: T.List[str]
        self.symbol_name_scoper = initialization.create_named_scope(sm.__scopes__)
        self.key_scoper = initialization.create_named_scope(self.__scopes__)

        # Fill with construction kwargs
        self.update(kwargs)

    # -------------------------------------------------------------------------
    # Dict API
    # -------------------------------------------------------------------------

    def keys(self):
        # type: () -> T.List[str]
        """
        An object providing a view on contained keys.
        """
        return self.dict.keys()

    def values(self):
        # type: () -> T.List[T.Any]
        """
        An object providing a view on contained values.
        """
        return self.dict.values()

    def items(self):
        # type: () -> T.List[T.Tuple[str, T.Any]]
        """
        An object providng a view on contained key/value pairs.
        """
        return self.dict.items()

    def get(self, key, default=None):
        # type: (str, T.Any) -> T.Any
        """
        Return the value for key if key is in the dictionary, else default.

        Args:
            key (str):
            default (any): Default value if key is not present.

        Returns:
            any: self[key] or default
        """
        return self.dict.get(key, default)

    def update(self, other):
        # type: (T.Union[Values, T.Mapping[str, T.Any]]) -> None
        """
        Updates keys in this Values from those of the other.
        """
        for key, value in other.items():
            self[key] = value

    def copy(self):
        # type: () -> Values
        """
        Returns a deepcopy of this Values.
        """
        return self.from_storage(self.to_storage())

    # -------------------------------------------------------------------------
    # Storage concept - see symforce.ops.storage_ops
    # -------------------------------------------------------------------------

    def storage_dim(self):
        # type: () -> int
        return sum([ops.StorageOps.storage_dim(v) for v in self.values()])

    def to_storage(self):
        # type: () -> T.List[T.Any]
        """
        Returns a flat list of unique values for every scalar element in this object.
        Equivalent to the list from :func:`flatten()`.
        """
        return self.flatten()[0]

    @classmethod
    def from_storage_index(cls, vector_values, indices):
        # type: (T.List[T.Scalar], T.Mapping[str, T.List[T.Any]]) -> Values
        """
        Takes a vectorized values and corresponding indices and reconstructs the original form.
        Reverse of :func:`to_storage()`.

        Args:
            vector_values (list): Vectorized values
            indices (dict(str, list)): Dict of key to the source (index, datatype, shape, item_index)
        """
        values = cls()
        for name, (inx, datatype, shape, item_index) in indices.items():
            vec = vector_values[inx : inx + cls._shape_to_dims(shape)]

            if datatype == "Scalar":
                values[name] = vec[0]
            elif datatype == "Matrix":
                values[name] = geo.Matrix(vec).reshape(*shape)
            elif datatype == "Values":
                values[name] = cls.from_storage_index(vec, item_index)
            elif datatype in {"Rot2", "Rot3", "Pose2", "Pose3", "Complex", "Quaternion"}:
                values[name] = getattr(geo, datatype).from_storage(vec)
            elif datatype in {"LinearCameraCal", "EquidistantEpipolarCameraCal", "ATANCameraCal"}:
                values[name] = getattr(cam, datatype).from_storage(vec)
            elif datatype == "np.ndarray":
                values[name] = np.array(vec).reshape(*shape)
            else:
                raise NotImplementedError('Unknown datatype: "{}"'.format(datatype))

        return values

    def from_storage(self, elements, index=None):
        # type: (T.List[T.Scalar], T.Dict[str, T.List[T.Any]]) -> Values
        if index is None:
            assert len(elements) == self.storage_dim()
            return Values.from_storage_index(elements, self.index())
        return Values.from_storage_index(elements, index)

    def symbolic(self, name, **kwargs):
        # type: (str, T.Dict) -> Values
        symbolic_values = Values()
        for k, v in self.items():
            symbolic_values[k] = ops.StorageOps.symbolic(v, "{}_{}".format(name, k), **kwargs)
        return symbolic_values

    def evalf(self):
        # type: () -> Values
        """
        Numerical evaluation.
        """
        vals, index = self.flatten()
        return self.from_storage_index([ops.StorageOps.evalf(e) for e in vals], index)

    # -------------------------------------------------------------------------
    # Group concept - see symforce.ops.group_ops
    # -------------------------------------------------------------------------

    def identity(self):
        # type: () -> Values
        identity_values = Values()
        for k, v in self.items():
            identity_values[k] = ops.GroupOps.identity(v)
        return identity_values

    def compose(self, other):
        # type: (Values) -> Values
        assert self.index() == other.index()
        composed_values = Values()
        for k, v in self.items():
            composed_values[k] = ops.GroupOps.compose(v, other[k])
        return composed_values

    def inverse(self):
        # type: () -> Values
        inverse_values = Values()
        for k, v in self.items():
            inverse_values[k] = ops.GroupOps.inverse(v)
        return inverse_values

    # -------------------------------------------------------------------------
    # Lie group concept - see symforce.ops.lie_group_ops
    # -------------------------------------------------------------------------

    def tangent_dim(self):
        # type: () -> int
        return sum([ops.LieGroupOps.tangent_dim(v) for v in self.values()])

    def from_tangent(self, vec, epsilon=0):
        # type: (T.List[T.Scalar], T.Scalar) -> Values
        updated_values = Values()
        inx = 0
        for k, v in self.items():
            dim = ops.LieGroupOps.tangent_dim(v)
            updated_values[k] = ops.LieGroupOps.from_tangent(v, vec[inx : inx + dim], epsilon)
            inx += dim
        return updated_values

    def to_tangent(self, epsilon=0):
        # type: (T.Scalar) -> T.List[T.Scalar]
        vec = []
        for v in self.values():
            vec.extend(ops.LieGroupOps.to_tangent(v, epsilon))
        return vec

    def storage_D_tangent(self):
        # type: () -> geo.Matrix
        storage_D_tangent = geo.Matrix(self.storage_dim(), self.tangent_dim()).zero()
        s_inx = 0
        t_inx = 0
        for v in self.values():
            s_dim = ops.StorageOps.storage_dim(v)
            t_dim = ops.LieGroupOps.tangent_dim(v)
            storage_D_tangent[
                s_inx : s_inx + s_dim, t_inx : t_inx + t_dim
            ] = ops.LieGroupOps.storage_D_tangent(v)
            s_inx += s_dim
            t_inx += t_dim
        return storage_D_tangent

    # -------------------------------------------------------------------------
    # Printing
    # -------------------------------------------------------------------------

    def format(self, indent=0):
        # type: (int) -> str
        """
        Pretty format as an indented tree.

        Args:
            indent (int): Number of spaces to indent

        Returns:
            str:
        """
        lines = []
        lines.append(self.__class__.__name__ + "(")
        for key, value in self.items():
            if isinstance(value, Values):
                value_str = value.format(indent=indent + 2)
            else:
                value_str = str(value)

            lines.append("  {}: {},".format(key, value_str))
        lines.append(")")
        indent_str = " " * indent
        return "\n".join(indent_str + line for line in lines)

    def __repr__(self):
        # type: () -> str
        """
        String representation, simply calls :func:`format()`.
        """
        return self.format()

    # -------------------------------------------------------------------------
    # Dict magic methods
    # -------------------------------------------------------------------------

    def _get_subvalues_and_key(self, key, create=False):
        # type: (str, bool) -> T.Tuple[Values, str]
        """
        Given a key, compute the full key name by applying name scopes and
        the innermost values that contains that key. Return the innermost values
        and the child key within that.

        Example:
            >>> v = Values(a=1, b=Values(c=3, d=Values(e=4, f=5)))
            >>> subvalues, key = v._get_subvalues_and_key('b.d.f')
            >>> assert subvalues == v['b.d']
            >>> assert key == 'f'

        Args:
            key (str):
            create (bool): If True, create inner Values along the way

        Returns:
            Values: Innermost sub-values containing key
            str: Key name within the values
        """
        # Prepend the key scopes if not the latest symbol scopes already
        key_scope_is_subset = sm.__scopes__[-len(self.__scopes__) :] == self.__scopes__
        if len(sm.__scopes__) > len(self.__scopes__) and key_scope_is_subset:
            full_key = key
        else:
            full_key = ".".join(self.__scopes__ + [key])

        split_key = full_key.split(".")
        key_path, key_name = split_key[:-1], split_key[-1]

        values = self
        for i, part in enumerate(key_path):
            if part not in values.dict:
                if not create:
                    return Values(), key_name
                values.dict[part] = Values()

            values = values.dict[part]
            assert isinstance(values, Values), 'Cannot set "{}", "{}" not a Values!'.format(
                full_key, ".".join(split_key[: i + 1])
            )

        return values, key_name

    def __getitem__(self, key):
        # type: (str) -> T.Any
        values, key_name = self._get_subvalues_and_key(key)
        return values.dict[key_name]

    def __setitem__(self, key, value):
        # type: (str, T.Any) -> None
        values, key_name = self._get_subvalues_and_key(key, create=True)
        values.dict[key_name] = value

    def __delitem__(self, key):
        # type: (str) -> None
        values, key_name = self._get_subvalues_and_key(key)
        del values.dict[key_name]

    def __contains__(self, key):
        # type: (str) -> bool
        values, key_name = self._get_subvalues_and_key(key)
        return values.dict.__contains__(key_name)

    # -------------------------------------------------------------------------
    # Name scope management
    # -------------------------------------------------------------------------

    @contextlib.contextmanager
    def scope(self, scope):
        # type: (str) -> T.Iterator[None]
        """
        Context manager to apply a name scope to both keys added to the values
        and new symbols created within the with block.
        """
        with self.symbol_name_scoper(scope), self.key_scoper(scope):
            yield None

    def _remove_scope(self, key):
        # type: (str) -> str
        """
        Strips the current Values scope off of the given key if present.
        """
        prefix = ".".join(self.__scopes__) + "."
        return key[key.startswith(prefix) and len(prefix) :]

    def add(self, value, **kwargs):
        # type: (T.Union[str, sm.Symbol], T.Any) -> None
        """
        Add a symbol into the values using its given name, either a Symbol or a string.
        Allows avoiding duplication of the sort `v['foo'] = sm.Symbol('foo')`.

        Args:
            value (Symbol or str):
        """
        if isinstance(value, sm.Symbol):
            self[self._remove_scope(value.name)] = value
        elif isinstance(value, str):
            symbol = sm.Symbol(value, **kwargs)
            self[self._remove_scope(symbol.name)] = symbol
        else:
            raise NameError("Expr of type {} has no .name".format(type(value)))

    @staticmethod
    def _shape_to_dims(shape):
        # type: (T.Sequence[int]) -> int
        """
        Compute the number of entries in an object of this shape.
        """
        return max(1, np.prod(shape))

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def flatten(self):
        # type: () -> T.Tuple[T.List[T.Any], T.Dict[str, T.List[T.Any]]]
        """
        Takes a dict of string keys to values and returns a flattened list of the values along
        with a dictionary from the original keys to their slices within the flattened vector.

        Return:
            list: Vectorized values
            dict(str, list): Dict of key to the source (index, datatype, shape, item_index)
        """
        inx = 0
        index_dict = collections.OrderedDict()
        vector_values = []  # type: T.List[T.Any]
        shape = tuple()  # type: T.Tuple
        for name, value in self.items():
            if isinstance(value, Values):
                datatype = "Values"
                values_list, item_index = value.flatten()
                shape = (len(values_list),)
                vec = geo.Matrix(values_list)
            elif isinstance(value, np.ndarray):
                datatype = "np.ndarray"
                shape = value.shape
                dim = int(np.prod(value.shape))
                vec = value.reshape(dim)
                item_index = {}
            elif isinstance(value, (sm.Expr, sm.Symbol, int, float)):
                datatype = "Scalar"
                shape = tuple()
                vec = geo.Matrix([value])
                item_index = {}
            elif hasattr(value, "shape"):
                datatype = "Matrix"
                shape = value.shape
                assert len(shape) > 0
                dim = int(np.prod(value.shape))
                vec = value.reshape(dim, 1)
                item_index = {}
            elif isinstance(value, Storage):
                datatype = value.__class__.__name__
                vec = geo.Matrix(value.to_storage())
                shape = (len(vec),)
                item_index = {}
            elif isinstance(value, (list, tuple)):
                datatype = "Matrix"
                vec = geo.Matrix(value)
                shape = vec.shape
                dim = int(np.prod(vec.shape))
                item_index = {}
            else:
                raise NotImplementedError(
                    'Unknown type: "{}" for key "{}"'.format(type(value), name)
                )

            vector_values.extend(vec)

            index_dict[name] = [inx, datatype, shape, item_index]

            inx += self._shape_to_dims(shape)

        return vector_values, index_dict

    def index(self):
        # type: () -> T.Dict[str, T.List[T.Any]]
        """
        Returns the index with structural information to reconstruct this values
        in :func:`from_storage()`. Equivalent to the index from :func:`flatten()`.
        """
        return self.flatten()[1]

    @staticmethod
    def _shape_implies_a_vector(shape):
        # type: (T.Tuple[int, int]) -> bool
        """
        Return True if the given shape is row or column vector-like.
        """
        return (
            len(shape) == 1
            or (len(shape) == 2 and shape[1] == 1)
            or (len(shape) == 2 and shape[0] == 1)
        )

    @classmethod
    def scalar_keys_recursive_from_index(cls, index):
        # type: (T.Mapping[str, T.Any]) -> T.List[str]
        """
        Compute a flat list of keys from the given values index. The order matches
        the serialized order of elements in the `.to_storage()` vector.
        """
        vec = []
        for name, (inx, datatype, shape, item_index) in index.items():
            if len(shape) == 0:
                vec.append(name)
            elif datatype == "Values":
                sub_vec = cls.scalar_keys_recursive_from_index(item_index)
                vec.extend("{}.{}".format(name, val) for val in sub_vec)
            elif cls._shape_implies_a_vector(shape) or len(shape) == 2:
                # Flatten row or column vectors to 1-D array
                vec.extend("{}[{}]".format(name, i) for i in range(cls._shape_to_dims(shape)))
            elif len(shape) == 2:
                # TODO(hayk): This isn't run currently because of the len(shape) == 2 check above.
                # Fix handling of matrices in codegen and re-enable this.
                vec.extend(
                    "{}[{},{}]".format(name, i, j) for i in range(shape[0]) for j in range(shape[1])
                )
            else:
                raise NotImplementedError()

        assert len(vec) == len(set(vec)), "Non-unique keys!\n{}".format(vec)
        return vec

    def items_recursive(self):
        # type: () -> T.List[T.Tuple[str, T.Any]]
        """
        Returns a flat list of key/value pairs for every element in this object.
        """
        flat_items = []
        for k, v in self.items():
            if isinstance(v, Values):
                dot_seperated_sub_keys = [
                    "{}.{}".format(k, sub_key) for sub_key in v.keys_recursive()
                ]
                flat_items.extend(zip(dot_seperated_sub_keys, v.values_recursive()))
            else:
                flat_items.append((k, v))
        return flat_items

    def keys_recursive(self):
        # type: () -> T.List[str]
        """
        Returns a flat list of unique keys for every element in this object.
        """
        return list(zip(*self.items_recursive()))[0]

    def values_recursive(self):
        # type: () -> T.List[T.Any]
        """
        Returns a flat list of elements stored in this Values object.
        """
        return list(zip(*self.items_recursive()))[1]

    # -------------------------------------------------------------------------
    # Miscellaneous helpers
    # -------------------------------------------------------------------------

    def __eq__(self, other):
        # type: (T.Any) -> bool
        """
        Exact equality check.
        """
        if isinstance(other, Values):
            return self.to_storage() == other.to_storage()
        else:
            return False

    def subs(self, *args, **kwargs):
        # type: (T.Any, T.Any) -> Values
        """
        Substitute given values of each scalar element into a new instance.
        """
        return self.from_storage([sm.S(s).subs(*args, **kwargs) for s in self.to_storage()])

    def simplify(self):
        # type: () -> Values
        """
        Simplify each scalar element into a new instance.
        """
        return self.from_storage(sm.simplify(sm.Matrix(self.to_storage())))


from symforce.ops.impl.class_lie_group_ops import ClassLieGroupOps

ops.StorageOps.register(Values, ClassLieGroupOps)
