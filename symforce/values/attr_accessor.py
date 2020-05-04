class AttrAccessor(object):
    """
    Helper to provide dot access for Values. This is an internal-only class.

    Example:
        >>> x0_dict = v['states.x0']
        >>> x0_attr = v.attr.states.x0
        >>> assert x0_dict == x0_attr
    """

    def __init__(self, values):
        """
        Construct by saving given values.

        Args:
            values (Values):
        """
        # Set this way because we're overriding __getattr__.
        self.__dict__["values"] = values

    def __getattr__(self, attr):
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

    def __setattr__(self, attr, value):
        """
        Set a key.

        Args:
            attr (str):
            value (any):
        """
        self.values[attr] = value

    def __dir__(self):
        """
        Enumerate the contained attributes, for introspection purposes like tab completion.

        Returns:
            iterable(str):
        """
        return self.__dict__["values"].keys()
