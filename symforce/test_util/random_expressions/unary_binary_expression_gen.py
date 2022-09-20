# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

import symforce.symbolic as sf
from symforce import logger
from symforce import typing as T
from symforce.test_util.random_expressions.op_probabilities import DEFAULT_BINARY_OPS
from symforce.test_util.random_expressions.op_probabilities import DEFAULT_LEAVES
from symforce.test_util.random_expressions.op_probabilities import DEFAULT_UNARY_OPS
from symforce.test_util.random_expressions.op_probabilities import OpProbability

# Import numba if installed.  This is not required, but significantly speeds up generate_D
try:
    from numba import njit

except ImportError:
    logger.warning(
        "Unable to import numba for UnaryBinaryExpressionGen; may be slow for large expressions"
    )

    def njit(f: T.Callable) -> T.Callable:
        return f


class UnaryBinaryExpressionGen:
    """
    Helper to generate random symbolic expressions composed of a set of given unary and
    binary operators. The user provides these operators, each with an assigned probability.
    Then they can sample expressions given a target number of operations.

    Takes care to sample uniformly from the space of all such possible expressions. This isn't
    necessarily important for many use cases, but seems as good as any single strategy, with the
    downside that the distributions become slow to compute for large expressions.

    Note that probabilities of unary and binary ops are completely independent. The sampling of
    unary vs binary ops is given by the variable D.

    Implementation reference (Appendix C):
        Deep Learning for Symbolic Mathematics (https://arxiv.org/abs/1912.01412)
    """

    def __init__(
        self,
        unary_ops: T.Sequence[OpProbability],
        binary_ops: T.Sequence[OpProbability],
        leaves: T.Sequence[sf.Scalar],
    ):
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.leaves = leaves

        self.ops = list(self.unary_ops) + list(self.binary_ops)
        self.ops_dict = {op.name: op for op in self.ops}

        self.unary_ops_probs = np.array([op.prob for op in self.unary_ops])
        self.unary_ops_probs = self.unary_ops_probs / sum(self.unary_ops_probs)

        self.binary_ops_probs = np.array([op.prob for op in self.binary_ops])
        self.binary_ops_probs = self.binary_ops_probs / sum(self.binary_ops_probs)

        # D[e][n] represents the number of different binary trees with n nodes
        # that can be generated from e empty nodes
        self.D: T.Optional[T.List[np.ndarray]] = None

    @staticmethod
    @njit
    def _next_row_of_D(
        num_leaves: int, max_ops: int, n: int, prev_row: np.ndarray, p1: float, p2: float
    ) -> np.ndarray:
        """
        Compute a row of D (actually D.T) from the previous row

        This is much faster when jitted with Numba, and we keep it outside of generate_D so it only
        has to be compiled once
        """
        s = np.zeros((2 * max_ops - n + 1))
        for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
            s[e] = num_leaves * s[e - 1] + p1 * prev_row[e] + p2 * prev_row[e + 1]
        return s

    @staticmethod
    def generate_D(
        max_ops: int, num_leaves: int = 1, p1: int = 1, p2: int = 1
    ) -> T.List[np.ndarray]:
        """
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(e, 0) = L ** e
            D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
        """
        # enumerate possible trees
        # first generate the tranposed version of D, then transpose it

        D = [np.array([0] + ([num_leaves ** e for e in range(1, 2 * max_ops + 1)]))]

        for n in range(1, max_ops + 1):  # number of operators
            D.append(UnaryBinaryExpressionGen._next_row_of_D(num_leaves, max_ops, n, D[-1], p1, p2))

        assert all(len(D[e]) >= len(D[e + 1]) for e in range(len(D) - 1))
        D_transpose = [
            np.array([D[e][n] for e in range(len(D)) if n < len(D[e])])
            for n in range(max(len(x) for x in D))
        ]

        return D_transpose

    def sample_next_pos(
        self, nb_empty: int, nb_ops: int, num_leaves: int = 1, p1: int = 1, p2: int = 1
    ) -> T.Tuple[int, int]:
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        assert self.D is not None

        probs: T.List[float] = []
        for i in range(nb_empty):
            probs.append((num_leaves ** i) * p1 * self.D[nb_empty - i][nb_ops - 1])
        for i in range(nb_empty):
            probs.append((num_leaves ** i) * p2 * self.D[nb_empty - i + 1][nb_ops - 1])

        np_probs = np.array([p / self.D[nb_empty][nb_ops] for p in probs], dtype=np.float64)

        e = np.random.choice(2 * nb_empty, p=np_probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty

        return e, arity

    def build_tree_sequence(self, num_ops_target: int) -> T.List:
        """
        Return a prefix notation sequence of the expression tree.
        """
        if self.D is None or num_ops_target >= len(self.D[0]) - 1:
            self.D = self.generate_D(num_ops_target)

        # Number of empty nodes
        e = 1

        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        stack = [None]

        for n in range(num_ops_target, 0, -1):
            k, arity = self.sample_next_pos(e, n, p1=1, p2=1)

            # The annotations in numpy are wrong, and don't include the Sequence[Any] overload
            if arity == 1:
                op = np.random.choice(self.unary_ops, p=self.unary_ops_probs)  # type: ignore
            else:
                op = np.random.choice(self.binary_ops, p=self.binary_ops_probs)  # type: ignore

            e += arity - 1 - k  # created empty nodes - skipped future leaves
            t_leaves += arity - 1  # update number of total leaves
            l_leaves += k  # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = stack[:pos] + [op.name] + [None for _ in range(arity)] + stack[pos + 1 :]

        # sanity check
        assert len([1 for v in stack if v in self.ops_dict]) == num_ops_target
        assert len([1 for v in stack if v is None]) == t_leaves

        # insert leaves into tree
        leaves = [np.random.choice(self.leaves) for _ in range(t_leaves)]
        for i in range(len(stack)):  # pylint: disable=consider-using-enumerate
            if stack[i] is None:
                stack[i] = leaves.pop()

        assert len(leaves) == 0

        return stack

    def seq_to_expr(self, seq: T.Sequence[T.Union[str, sf.Scalar]]) -> sf.Expr:
        """
        Convert a prefix notation sequence into a sympy expression.
        """

        def _seq_to_expr(
            seq: T.Sequence[T.Union[str, sf.Scalar]]
        ) -> T.Tuple[sf.Scalar, T.Sequence[T.Union[str, sf.Scalar]]]:
            assert len(seq) > 0
            t = seq[0]
            if t in self.ops_dict:
                op = self.ops_dict[T.cast(str, t)]
                args = []
                l1 = seq[1:]
                for _ in range(op.arity):
                    i1, l1 = _seq_to_expr(l1)
                    args.append(i1)
                return op.func(*args), l1
            elif t in self.leaves:
                return T.cast(sf.Scalar, t), seq[1:]
            else:
                assert f"Unknown: {t}"
                return 0, []  # Just for mypy..

        return _seq_to_expr(seq)[0]

    def build_expr(self, num_ops_target: int) -> sf.Scalar:
        """
        Return an expression with the given op target.
        """
        seq = self.build_tree_sequence(num_ops_target=num_ops_target)
        return self.seq_to_expr(seq)

    def build_expr_vec(self, num_ops_target: int, num_exprs: int = None) -> sf.M:
        """
        Return a vector of expressions with the total given op target. If no num_exprs
        is provided, uses an approximate square root of the num_ops_target.
        """
        # Empirical fudge factor for simplifications
        num_ops_target = int(1.1 * num_ops_target)

        if num_exprs is None:
            num_exprs = max(1, int(np.sqrt(num_ops_target)))
        target_per_expr = int(num_ops_target / num_exprs)

        exprs: T.List[sf.Scalar] = []
        while len(exprs) < num_exprs:
            try:
                exprs.append(self.build_expr(target_per_expr))
            except (ZeroDivisionError, RuntimeError) as e:
                print(e)
                print("Skipping.")

        return sf.M(exprs)

    @classmethod
    def default(
        cls,
        unary_ops: T.Sequence[OpProbability] = DEFAULT_UNARY_OPS,
        binary_ops: T.Sequence[OpProbability] = DEFAULT_BINARY_OPS,
        leaves: T.Sequence[sf.Scalar] = DEFAULT_LEAVES,
    ) -> UnaryBinaryExpressionGen:
        """
        Construct with a reasonable default op distribution.
        """
        return cls(unary_ops=unary_ops, binary_ops=binary_ops, leaves=leaves)
