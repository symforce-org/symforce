# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from symforce import typing as T


class AttrAccessor:
    """
    Helper to provide dot access for Values. This is an internal-only class.

    Example:
        >>> x0_dict = v['states.x0']
        >>> x0_attr = v.attr.states.x0
        >>> assert x0_dict == x0_attr
    """

    def __init__(self, values: T.Dict[str, T.Any]) -> None:
        """
        Construct by saving given values.

        Args:
            values (Values):
        """
        # Set this way because we're overriding __getattr__.
        self.__dict__["values"] = values

    def __getattr__(self, attr: str) -> T.Any:
        """
        Access a key with the given path.

        Args:
            attr (str): Example, 'states.x0'

        Returns:
            any:
        """
        value = self.values[attr]
        if type(value).__name__ == "Values":
            # To allow chaining, return the attr for sub-values.
            return value.attr
        else:
            # Otherwise just return the values.
            return self.values[attr]

    def __setattr__(self, attr: str, value: T.Any) -> None:
        """
        Set a key.

        Args:
            attr (str):
            value (any):
        """
        self.values[attr] = value

    def __dir__(self) -> T.List[str]:
        """
        Enumerate the contained attributes, for introspection purposes like tab completion.

        Returns:
            iterable(str):
        """
        return list(self.__dict__["values"].keys())
