from cython.operator cimport dereference as deref, preincrement as inc
cimport symengine
from symengine cimport (RCP, pair, map_basic_basic, umap_int_basic,
    umap_int_basic_iterator, umap_basic_num, umap_basic_num_iterator,
    rcp_const_basic, std_pair_short_rcp_const_basic,
    rcp_const_seriescoeffinterface, CRCPBasic)
from libcpp cimport bool as cppbool
from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF, \
    PyObject_CallMethodObjArgs
from libc.string cimport memcpy
import cython
import itertools
import numbers
from operator import mul
from functools import reduce
import collections
import warnings
from symengine.utilities import is_sequence
import os
import sys
from cpython.pycapsule cimport PyCapsule_GetPointer
from collections.abc import MutableMapping

try:
    import numpy as np
    # Lambdify requires NumPy (since b713a61, see gh-112)
    have_numpy = True
except ImportError:
    have_numpy = False

include "config.pxi"

class SympifyError(Exception):
    pass

cpdef object capsule_to_basic(object capsule):
    cdef CRCPBasic *p = <CRCPBasic*>PyCapsule_GetPointer(capsule, NULL)
    return c2py(p.m)

cpdef void assign_to_capsule(object capsule, object value):
    cdef CRCPBasic *p_cap = <CRCPBasic*>PyCapsule_GetPointer(capsule, NULL)
    cdef Basic v = sympify(value)
    p_cap.m = v.thisptr

cdef object c2py(rcp_const_basic o):
    cdef Basic r
    if (symengine.is_a_Add(deref(o))):
        r = Expr.__new__(Add)
    elif (symengine.is_a_Mul(deref(o))):
        r = Expr.__new__(Mul)
    elif (symengine.is_a_Pow(deref(o))):
        r = Expr.__new__(Pow)
    elif (symengine.is_a_Integer(deref(o))):
        if (deref(symengine.rcp_static_cast_Integer(o)).is_zero()):
            return S.Zero
        elif (deref(symengine.rcp_static_cast_Integer(o)).is_one()):
            return S.One
        elif (deref(symengine.rcp_static_cast_Integer(o)).is_minus_one()):
            return S.NegativeOne
        r = Number.__new__(Integer)
    elif (symengine.is_a_Rational(deref(o))):
        r = S.Half
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.Half
        r = Number.__new__(Rational)
    elif (symengine.is_a_Complex(deref(o))):
        r = S.ImaginaryUnit
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.ImaginaryUnit
        r = Complex.__new__(Complex)
    elif (symengine.is_a_Dummy(deref(o))):
        r = Symbol.__new__(Dummy)
    elif (symengine.is_a_Symbol(deref(o))):
        if (symengine.is_a_PySymbol(deref(o))):
            return <object>(deref(symengine.rcp_static_cast_PySymbol(o)).get_py_object())
        r = Symbol.__new__(Symbol)
    elif (symengine.is_a_Constant(deref(o))):
        r = S.Pi
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.Pi
        r = S.Exp1
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.Exp1
        r = S.GoldenRatio
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.GoldenRatio
        r = S.Catalan
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.Catalan
        r = S.EulerGamma
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.EulerGamma
        r = Constant.__new__(Constant)
    elif (symengine.is_a_Infty(deref(o))):
        if (deref(symengine.rcp_static_cast_Infty(o)).is_positive()):
            return S.Infinity
        elif (deref(symengine.rcp_static_cast_Infty(o)).is_negative()):
            return S.NegativeInfinity
        return S.ComplexInfinity
    elif (symengine.is_a_NaN(deref(o))):
        return S.NaN
    elif (symengine.is_a_PyFunction(deref(o))):
        r = PyFunction.__new__(PyFunction)
    elif (symengine.is_a_FunctionSymbol(deref(o))):
        r = FunctionSymbol.__new__(FunctionSymbol)
    elif (symengine.is_a_Abs(deref(o))):
        r = Function.__new__(Abs)
    elif (symengine.is_a_Max(deref(o))):
        r = Function.__new__(Max)
    elif (symengine.is_a_Min(deref(o))):
        r = Function.__new__(Min)
    elif (symengine.is_a_BooleanAtom(deref(o))):
        if (deref(symengine.rcp_static_cast_BooleanAtom(o)).get_val()):
            return S.true
        return S.false
    elif (symengine.is_a_Equality(deref(o))):
        r = Relational.__new__(Equality)
    elif (symengine.is_a_Unequality(deref(o))):
        r = Relational.__new__(Unequality)
    elif (symengine.is_a_LessThan(deref(o))):
        r = Relational.__new__(LessThan)
    elif (symengine.is_a_StrictLessThan(deref(o))):
        r = Relational.__new__(StrictLessThan)
    elif (symengine.is_a_Gamma(deref(o))):
        r = Function.__new__(Gamma)
    elif (symengine.is_a_Derivative(deref(o))):
        r = Expr.__new__(Derivative)
    elif (symengine.is_a_Subs(deref(o))):
        r = Expr.__new__(Subs)
    elif (symengine.is_a_RealDouble(deref(o))):
        r = Number.__new__(RealDouble)
    elif (symengine.is_a_ComplexDouble(deref(o))):
        r = ComplexDouble.__new__(ComplexDouble)
    elif (symengine.is_a_RealMPFR(deref(o))):
        r = Number.__new__(RealMPFR)
    elif (symengine.is_a_ComplexMPC(deref(o))):
        r = ComplexMPC.__new__(ComplexMPC)
    elif (symengine.is_a_Log(deref(o))):
        r = Function.__new__(Log)
    elif (symengine.is_a_Sin(deref(o))):
        r = Function.__new__(Sin)
    elif (symengine.is_a_Cos(deref(o))):
        r = Function.__new__(Cos)
    elif (symengine.is_a_Tan(deref(o))):
        r = Function.__new__(Tan)
    elif (symengine.is_a_Cot(deref(o))):
        r = Function.__new__(Cot)
    elif (symengine.is_a_Csc(deref(o))):
        r = Function.__new__(Csc)
    elif (symengine.is_a_Sec(deref(o))):
        r = Function.__new__(Sec)
    elif (symengine.is_a_ASin(deref(o))):
        r = Function.__new__(ASin)
    elif (symengine.is_a_ACos(deref(o))):
        r = Function.__new__(ACos)
    elif (symengine.is_a_ATan(deref(o))):
        r = Function.__new__(ATan)
    elif (symengine.is_a_ACot(deref(o))):
        r = Function.__new__(ACot)
    elif (symengine.is_a_ACsc(deref(o))):
        r = Function.__new__(ACsc)
    elif (symengine.is_a_ASec(deref(o))):
        r = Function.__new__(ASec)
    elif (symengine.is_a_Sinh(deref(o))):
        r = Function.__new__(Sinh)
    elif (symengine.is_a_Cosh(deref(o))):
        r = Function.__new__(Cosh)
    elif (symengine.is_a_Tanh(deref(o))):
        r = Function.__new__(Tanh)
    elif (symengine.is_a_Coth(deref(o))):
        r = Function.__new__(Coth)
    elif (symengine.is_a_Csch(deref(o))):
        r = Function.__new__(Csch)
    elif (symengine.is_a_Sech(deref(o))):
        r = Function.__new__(Sech)
    elif (symengine.is_a_ASinh(deref(o))):
        r = Function.__new__(ASinh)
    elif (symengine.is_a_ACosh(deref(o))):
        r = Function.__new__(ACosh)
    elif (symengine.is_a_ATanh(deref(o))):
        r = Function.__new__(ATanh)
    elif (symengine.is_a_ACoth(deref(o))):
        r = Function.__new__(ACoth)
    elif (symengine.is_a_ACsch(deref(o))):
        r = Function.__new__(ACsch)
    elif (symengine.is_a_ASech(deref(o))):
        r = Function.__new__(ASech)
    elif (symengine.is_a_ATan2(deref(o))):
        r = Function.__new__(ATan2)
    elif (symengine.is_a_LambertW(deref(o))):
        r = Function.__new__(LambertW)
    elif (symengine.is_a_Zeta(deref(o))):
        r = Function.__new__(zeta)
    elif (symengine.is_a_DirichletEta(deref(o))):
        r = Function.__new__(dirichlet_eta)
    elif (symengine.is_a_KroneckerDelta(deref(o))):
        r = Function.__new__(KroneckerDelta)
    elif (symengine.is_a_LeviCivita(deref(o))):
        r = Function.__new__(LeviCivita)
    elif (symengine.is_a_Erf(deref(o))):
        r = Function.__new__(erf)
    elif (symengine.is_a_Erfc(deref(o))):
        r = Function.__new__(erfc)
    elif (symengine.is_a_LowerGamma(deref(o))):
        r = Function.__new__(lowergamma)
    elif (symengine.is_a_UpperGamma(deref(o))):
        r = Function.__new__(uppergamma)
    elif (symengine.is_a_LogGamma(deref(o))):
        r = Function.__new__(loggamma)
    elif (symengine.is_a_Beta(deref(o))):
        r = Function.__new__(beta)
    elif (symengine.is_a_PolyGamma(deref(o))):
        r = Function.__new__(polygamma)
    elif (symengine.is_a_Sign(deref(o))):
        r = Function.__new__(sign)
    elif (symengine.is_a_Floor(deref(o))):
        r = Function.__new__(floor)
    elif (symengine.is_a_Ceiling(deref(o))):
        r = Function.__new__(ceiling)
    elif (symengine.is_a_Conjugate(deref(o))):
        r = Function.__new__(conjugate)
    elif (symengine.is_a_PyNumber(deref(o))):
        r = PyNumber.__new__(PyNumber)
    elif (symengine.is_a_Piecewise(deref(o))):
        r = Function.__new__(Piecewise)
    elif (symengine.is_a_Contains(deref(o))):
        r = Boolean.__new__(Contains)
    elif (symengine.is_a_Interval(deref(o))):
        r = Set.__new__(Interval)
    elif (symengine.is_a_EmptySet(deref(o))):
        r = Set.__new__(EmptySet)
    elif (symengine.is_a_Reals(deref(o))):
        r = Set.__new__(Reals)
    elif (symengine.is_a_Integers(deref(o))):
        r = Set.__new__(Integers)
    elif (symengine.is_a_UniversalSet(deref(o))):
        r = Set.__new__(UniversalSet)
    elif (symengine.is_a_FiniteSet(deref(o))):
        r = Set.__new__(FiniteSet)
    elif (symengine.is_a_Union(deref(o))):
        r = Set.__new__(Union)
    elif (symengine.is_a_Complement(deref(o))):
        r = Set.__new__(Complement)
    elif (symengine.is_a_ConditionSet(deref(o))):
        r = Set.__new__(ConditionSet)
    elif (symengine.is_a_ImageSet(deref(o))):
        r = Set.__new__(ImageSet)
    elif (symengine.is_a_And(deref(o))):
        r = Boolean.__new__(And)
    elif (symengine.is_a_Not(deref(o))):
        r = Boolean.__new__(Not)
    elif (symengine.is_a_Or(deref(o))):
        r = Boolean.__new__(Or)
    elif (symengine.is_a_Xor(deref(o))):
        r = Boolean.__new__(Xor)
    elif (symengine.is_a_UnevaluatedExpr(deref(o))):
        r = Function.__new__(UnevaluatedExpr)
    else:
        raise Exception("Unsupported SymEngine class.")
    r.thisptr = o
    return r

def sympy2symengine(a, raise_error=False):
    """
    Converts 'a' from SymPy to SymEngine.

    If the expression cannot be converted, it either returns None (if
    raise_error==False) or raises a SympifyError exception (if
    raise_error==True).
    """
    import sympy
    from sympy.core.function import AppliedUndef as sympy_AppliedUndef
    if isinstance(a, sympy.Symbol):
        return Symbol(a.name)
    elif isinstance(a, sympy.Dummy):
        return Dummy(a.name)
    elif isinstance(a, sympy.Mul):
        return mul(*[sympy2symengine(x, raise_error) for x in a.args])
    elif isinstance(a, sympy.Add):
        return add(*[sympy2symengine(x, raise_error) for x in a.args])
    elif isinstance(a, (sympy.Pow, sympy.exp)):
        x, y = a.as_base_exp()
        return sympy2symengine(x, raise_error) ** sympy2symengine(y, raise_error)
    elif isinstance(a, sympy.Integer):
        return Integer(a.p)
    elif isinstance(a, sympy.Rational):
        return Integer(a.p) / Integer(a.q)
    elif isinstance(a, sympy.Float):
        IF HAVE_SYMENGINE_MPFR:
            if a._prec > 53:
                return RealMPFR(str(a), a._prec)
            else:
                return RealDouble(float(str(a)))
        ELSE:
            return RealDouble(float(str(a)))
    elif a is sympy.I:
        return I
    elif a is sympy.E:
        return E
    elif a is sympy.pi:
        return pi
    elif a is sympy.GoldenRatio:
        return golden_ratio
    elif a is sympy.Catalan:
        return catalan
    elif a is sympy.EulerGamma:
        return eulergamma
    elif a is sympy.S.NegativeInfinity:
        return minus_oo
    elif a is sympy.S.Infinity:
        return oo
    elif a is sympy.S.ComplexInfinity:
        return zoo
    elif a is sympy.nan:
        return nan
    elif a is sympy.S.true:
        return true
    elif a is sympy.S.false:
        return false
    elif isinstance(a, sympy.functions.elementary.trigonometric.TrigonometricFunction):
        if isinstance(a, sympy.sin):
            return sin(a.args[0])
        elif isinstance(a, sympy.cos):
            return cos(a.args[0])
        elif isinstance(a, sympy.tan):
            return tan(a.args[0])
        elif isinstance(a, sympy.cot):
            return cot(a.args[0])
        elif isinstance(a, sympy.csc):
            return csc(a.args[0])
        elif isinstance(a, sympy.sec):
            return sec(a.args[0])
    elif isinstance(a, sympy.functions.elementary.trigonometric.InverseTrigonometricFunction):
        if isinstance(a, sympy.asin):
            return asin(a.args[0])
        elif isinstance(a, sympy.acos):
            return acos(a.args[0])
        elif isinstance(a, sympy.atan):
            return atan(a.args[0])
        elif isinstance(a, sympy.acot):
            return acot(a.args[0])
        elif isinstance(a, sympy.acsc):
            return acsc(a.args[0])
        elif isinstance(a, sympy.asec):
            return asec(a.args[0])
        elif isinstance(a, sympy.atan2):
            return atan2(*a.args)
    elif isinstance(a, sympy.functions.elementary.hyperbolic.HyperbolicFunction):
        if isinstance(a, sympy.sinh):
            return sinh(a.args[0])
        elif isinstance(a, sympy.cosh):
            return cosh(a.args[0])
        elif isinstance(a, sympy.tanh):
            return tanh(a.args[0])
        elif isinstance(a, sympy.coth):
            return coth(a.args[0])
        elif isinstance(a, sympy.csch):
            return csch(a.args[0])
        elif isinstance(a, sympy.sech):
            return sech(a.args[0])
    elif isinstance(a, sympy.asinh):
        return asinh(a.args[0])
    elif isinstance(a, sympy.acosh):
        return acosh(a.args[0])
    elif isinstance(a, sympy.atanh):
        return atanh(a.args[0])
    elif isinstance(a, sympy.acoth):
        return acoth(a.args[0])
    elif isinstance(a, sympy.log):
        return log(a.args[0])
    elif isinstance(a, sympy.Abs):
        return abs(sympy2symengine(a.args[0], raise_error))
    elif isinstance(a, sympy.Max):
        return _max(*a.args)
    elif isinstance(a, sympy.Min):
        return _min(*a.args)
    elif isinstance(a, sympy.Equality):
        return eq(*a.args)
    elif isinstance(a, sympy.Unequality):
        return ne(*a.args)
    elif isinstance(a, sympy.GreaterThan):
        return ge(*a.args)
    elif isinstance(a, sympy.StrictGreaterThan):
        return gt(*a.args)
    elif isinstance(a, sympy.LessThan):
        return le(*a.args)
    elif isinstance(a, sympy.StrictLessThan):
        return lt(*a.args)
    elif isinstance(a, sympy.LambertW):
        return LambertW(a.args[0])
    elif isinstance(a, sympy.zeta):
        return zeta(*a.args)
    elif isinstance(a, sympy.dirichlet_eta):
        return dirichlet_eta(a.args[0])
    elif isinstance(a, sympy.KroneckerDelta):
        return KroneckerDelta(*a.args)
    elif isinstance(a, sympy.LeviCivita):
        return LeviCivita(*a.args)
    elif isinstance(a, sympy.erf):
        return erf(a.args[0])
    elif isinstance(a, sympy.erfc):
        return erfc(a.args[0])
    elif isinstance(a, sympy.lowergamma):
        return lowergamma(*a.args)
    elif isinstance(a, sympy.uppergamma):
        return uppergamma(*a.args)
    elif isinstance(a, sympy.loggamma):
        return loggamma(a.args[0])
    elif isinstance(a, sympy.beta):
        return beta(*a.args)
    elif isinstance(a, sympy.polygamma):
        return polygamma(*a.args)
    elif isinstance(a, sympy.sign):
        return sign(a.args[0])
    elif isinstance(a, sympy.floor):
        return floor(a.args[0])
    elif isinstance(a, sympy.ceiling):
        return ceiling(a.args[0])
    elif isinstance(a, sympy.conjugate):
        return conjugate(a.args[0])
    elif isinstance(a, sympy.And):
        return logical_and(*a.args)
    elif isinstance(a, sympy.Or):
        return logical_or(*a.args)
    elif isinstance(a, sympy.Not):
        return logical_not(a.args[0])
    elif isinstance(a, sympy.Nor):
        return Nor(*a.args)
    elif isinstance(a, sympy.Nand):
        return Nand(*a.args)
    elif isinstance(a, sympy.Xor):
        return logical_xor(*a.args)
    elif isinstance(a, sympy.gamma):
        return gamma(a.args[0])
    elif isinstance(a, sympy.Derivative):
        return Derivative(a.expr, *a.variables)
    elif isinstance(a, sympy.Subs):
        return Subs(a.expr, a.variables, a.point)
    elif isinstance(a, sympy_AppliedUndef):
        name = str(a.func)
        return function_symbol(name, *(a.args))
    elif isinstance(a, (sympy.Piecewise)):
        return piecewise(*(a.args))
    elif a is sympy.S.Reals:
        return S.Reals
    elif a is sympy.S.Integers:
        return S.Integers
    elif isinstance(a, sympy.Interval):
        return interval(*(a.args))
    elif a is sympy.S.EmptySet:
        return S.EmptySet
    elif a is sympy.S.UniversalSet:
        return S.UniversalSet
    elif isinstance(a, sympy.FiniteSet):
        return finiteset(*(a.args))
    elif isinstance(a, sympy.Contains):
        return contains(*(a.args))
    elif isinstance(a, sympy.Union):
        return set_union(*(a.args))
    elif isinstance(a, sympy.Intersection):
        return set_intersection(*(a.args))
    elif isinstance(a, sympy.Complement):
        return set_complement(*(a.args))
    elif isinstance(a, sympy.ImageSet):
        return imageset(*(a.args))
    elif isinstance(a, sympy.Function):
        return PyFunction(a, a.args, a.func, sympy_module)
    elif isinstance(a, sympy.UnevaluatedExpr):
        return UnevaluatedExpr(a.args[0])
    elif isinstance(a, sympy.MatrixBase):
        row, col = a.shape
        v = []
        for r in a.tolist():
            for e in r:
                v.append(e)
        if isinstance(a, sympy.MutableDenseMatrix):
            return MutableDenseMatrix(row, col, v)
        elif isinstance(a, sympy.ImmutableDenseMatrix):
            return ImmutableDenseMatrix(row, col, v)
        else:
            raise NotImplementedError
    elif isinstance(a, sympy.polys.domains.modularinteger.ModularInteger):
        return PyNumber(a, sympy_module)
    elif sympy.__version__ > '1.0':
        if isinstance(a, sympy.acsch):
            return acsch(a.args[0])
        elif isinstance(a, sympy.asech):
            return asech(a.args[0])
        elif isinstance(a, sympy.ConditionSet):
            return conditionset(*(a.args))

    if raise_error:
        raise SympifyError(("sympy2symengine: Cannot convert '%r' (of type %s)"
                            " to a symengine type.") % (a, type(a)))


def sympify(a):
    """
    Converts an expression 'a' into a SymEngine type.

    Arguments
    =========

    a ............. An expression to convert.

    Examples
    ========

    >>> from symengine import sympify
    >>> sympify(1)
    1
    >>> sympify("a+b")
    a + b
    """
    if isinstance(a, str):
        return c2py(symengine.parse(a.encode("utf-8")))
    elif isinstance(a, tuple):
        v = []
        for e in a:
            v.append(sympify(e))
        return tuple(v)
    elif isinstance(a, list):
        v = []
        for e in a:
            v.append(sympify(e))
        return v
    return _sympify(a, True)


def _sympify(a, raise_error=True):
    """
    Converts an expression 'a' into a SymEngine type.

    Arguments
    =========

    a ............. An expression to convert.
    raise_error ... Will raise an error on a failure (default True), otherwise
                    it returns None if 'a' cannot be converted.

    Examples
    ========

    >>> from symengine.li.symengine_wrapper import _sympify
    >>> _sympify(1)
    1
    >>> _sympify("abc", False)
    >>>

    """
    if isinstance(a, (Basic, MatrixBase)):
        return a
    elif isinstance(a, bool):
        return (true if a else false)
    elif isinstance(a, numbers.Integral):
        return Integer(a)
    elif isinstance(a, float):
        return RealDouble(a)
    elif isinstance(a, complex):
        return ComplexDouble(a)
    elif hasattr(a, '_symengine_'):
        return _sympify(a._symengine_(), raise_error)
    elif hasattr(a, '_sympy_'):
        return _sympify(a._sympy_(), raise_error)
    elif hasattr(a, 'pyobject'):
        return _sympify(a.pyobject(), raise_error)

    try:
        import sympy
        return sympy2symengine(a, raise_error)
    except ImportError:
        pass

    if raise_error:
        raise SympifyError(
            "sympify: Cannot convert '%r' (of type %s) to a symengine type." % (
                a, type(a)))

funcs = {}

def get_function_class(function, module):
    if not function in funcs:
        funcs[function] = PyFunctionClass(function, module)
    return funcs[function]

class Singleton(object):

    __call__ = staticmethod(sympify)

    @property
    def Zero(self):
        return zero

    @property
    def One(self):
        return one

    @property
    def NegativeOne(self):
        return minus_one

    @property
    def Half(self):
        return half

    @property
    def Pi(self):
        return pi

    @property
    def NaN(self):
        return nan

    @property
    def Infinity(self):
        return oo

    @property
    def NegativeInfinity(self):
        return minus_oo

    @property
    def ComplexInfinity(self):
        return zoo

    @property
    def Exp1(self):
        return E

    @property
    def GoldenRatio(self):
        return golden_ratio

    @property
    def Catalan(self):
        return catalan

    @property
    def EulerGamma(self):
        return eulergamma

    @property
    def ImaginaryUnit(self):
        return I

    @property
    def true(self):
        return true

    @property
    def false(self):
        return false

    @property
    def EmptySet(self):
        return empty_set_singleton

    @property
    def UniversalSet(self):
        return universal_set_singleton

    @property
    def Integers(self):
        return integers_singleton

    @property
    def Reals(self):
        return reals_singleton

S = Singleton()


cdef class DictBasicIter(object):

    cdef init(self, map_basic_basic.iterator begin, map_basic_basic.iterator end):
        self.begin = begin
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.begin != self.end:
            obj = c2py(deref(self.begin).first)
        else:
            raise StopIteration
        inc(self.begin)
        return obj


cdef class _DictBasic(object):

    def __init__(self, tocopy = None):
        if tocopy != None:
            self.add_dict(tocopy)

    def as_dict(self):
        ret = {}
        it = self.c.begin()
        while it != self.c.end():
            ret[c2py(deref(it).first)] = c2py(deref(it).second)
            inc(it)
        return ret

    def add_dict(self, d):
        cdef _DictBasic D
        if isinstance(d, DictBasic):
            D = d
            self.c.insert(D.c.begin(), D.c.end())
        else:
            for key, value in d.iteritems():
                self.add(key, value)

    def add(self, key, value):
        cdef Basic K = sympify(key)
        cdef Basic V = sympify(value)
        cdef symengine.std_pair_rcp_const_basic_rcp_const_basic pair
        pair.first = K.thisptr
        pair.second = V.thisptr
        return self.c.insert(pair).second

    def copy(self):
        return DictBasic(self)

    __copy__ = copy

    def __len__(self):
        return self.c.size()

    def __getitem__(self, key):
        cdef Basic K = sympify(key)
        it = self.c.find(K.thisptr)
        if it == self.c.end():
            raise KeyError(key)
        else:
            return c2py(deref(it).second)

    def __setitem__(self, key, value):
        cdef Basic K = sympify(key)
        cdef Basic V = sympify(value)
        self.c[K.thisptr] = V.thisptr

    def clear(self):
        self.clear()

    def __delitem__(self, key):
        cdef Basic K = sympify(key)
        self.c.erase(K.thisptr)

    def __contains__(self, key):
        cdef Basic K = sympify(key)
        it = self.c.find(K.thisptr)
        return it != self.c.end()

    def __iter__(self):
        cdef DictBasicIter d = DictBasicIter()
        d.init(self.c.begin(), self.c.end())
        return d


class DictBasic(_DictBasic, MutableMapping):

    def __str__(self):
        return "{" + ", ".join(["%s: %s" % (str(key), str(value)) for key, value in self.items()]) + "}"

    def __repr__(self):
        return self.__str__()

def get_dict(*args):
    """
    Returns a DictBasic instance from args. Inputs can be,
        1. a DictBasic
        2. a Python dictionary
        3. two args old, new
    """
    cdef _DictBasic D = DictBasic()
    if len(args) == 2:
        if is_sequence(args[0]):
            for k, v in zip(args[0], args[1]):
                D.add(k, v)
        else:
            D.add(args[0], args[1])
        return D
    elif len(args) == 1:
        arg = args[0]
    else:
        raise TypeError("subs/msubs takes one or two arguments (%d given)" % \
                len(args))
    if isinstance(arg, DictBasic):
        return arg
    for k, v in arg.items():
        D.add(k, v)
    return D


cdef tuple vec_basic_to_tuple(symengine.vec_basic& vec):
    return tuple(vec_basic_to_list(vec))


cdef list vec_basic_to_list(symengine.vec_basic& vec):
    result = []
    for i in range(vec.size()):
        result.append(c2py(<rcp_const_basic>(vec[i])))
    return result


cdef list vec_pair_to_list(symengine.vec_pair& vec):
    result = []
    cdef rcp_const_basic a, b
    for i in range(vec.size()):
        a = <rcp_const_basic>vec[i].first
        b = <rcp_const_basic>vec[i].second
        result.append((c2py(a), c2py(b)))
    return result


repr_latex=[False]

cdef class Basic(object):

    def __str__(self):
        return deref(self.thisptr).__str__().decode("utf-8")

    def __repr__(self):
        return self.__str__()

    def _repr_latex_(self):
        if repr_latex[0]:
            return "${}$".format(latex(self))
        else:
            return None

    def __hash__(self):
        return deref(self.thisptr).hash()

    def __dealloc__(self):
        self.thisptr.reset()

    def _unsafe_reset(self):
        self.thisptr.reset()

    def __add__(a, b):
        cdef Basic A = _sympify(a, False)
        B_ = _sympify(b, False)
        if A is None or B_ is None or isinstance(B_, MatrixBase): return NotImplemented
        cdef Basic B = B_
        return c2py(symengine.add(A.thisptr, B.thisptr))

    def __sub__(a, b):
        cdef Basic A = _sympify(a, False)
        B_ = _sympify(b, False)
        if A is None or B_ is None or isinstance(B_, MatrixBase): return NotImplemented
        cdef Basic B = B_
        return c2py(symengine.sub(A.thisptr, B.thisptr))

    def __mul__(a, b):
        cdef Basic A = _sympify(a, False)
        B_ = _sympify(b, False)
        if A is None or B_ is None or isinstance(B_, MatrixBase): return NotImplemented
        cdef Basic B = B_
        return c2py(symengine.mul(A.thisptr, B.thisptr))

    def __truediv__(a, b):
        cdef Basic A = _sympify(a, False)
        cdef Basic B = _sympify(b, False)
        if A is None or B is None: return NotImplemented
        return c2py(symengine.div(A.thisptr, B.thisptr))

    def __pow__(a, b, c):
        if c is not None:
            return powermod(a, b, c)
        cdef Basic A = _sympify(a, False)
        cdef Basic B = _sympify(b, False)
        if A is None or B is None: return NotImplemented
        return c2py(symengine.pow(A.thisptr, B.thisptr))

    def __neg__(Basic self not None):
        return c2py(symengine.neg(self.thisptr))

    def __abs__(Basic self not None):
        return c2py(symengine.abs(self.thisptr))

    def __richcmp__(a, b, int op):
        A = _sympify(a, False)
        B = _sympify(b, False)
        if not (isinstance(A, Basic) and isinstance(B, Basic)):
            if (op == 2):
                return False
            elif (op == 3):
                return True
            else:
                return NotImplemented
        return Basic._richcmp_(A, B, op)

    def _richcmp_(Basic A, Basic B, int op):
        if (op == 2):
            return symengine.eq(deref(A.thisptr), deref(B.thisptr))
        elif (op == 3):
            return symengine.neq(deref(A.thisptr), deref(B.thisptr))
        if (op == 0):
            return c2py(<rcp_const_basic>(symengine.Lt(A.thisptr, B.thisptr)))
        elif (op == 1):
            return c2py(<rcp_const_basic>(symengine.Le(A.thisptr, B.thisptr)))
        elif (op == 4):
            return c2py(<rcp_const_basic>(symengine.Gt(A.thisptr, B.thisptr)))
        elif (op == 5):
            return c2py(<rcp_const_basic>(symengine.Ge(A.thisptr, B.thisptr)))

    def expand(Basic self not None, cppbool deep=True):
        return c2py(symengine.expand(self.thisptr, deep))

    def _diff(Basic self not None, Basic x):
        return c2py(symengine.diff(self.thisptr, x.thisptr))

    def diff(self, *args):
        if len(args) == 0:
            f = self.free_symbols
            if (len(f) != 1):
                raise RuntimeError("Variable w.r.t should be given")
            return self._diff(f.pop())
        return diff(self, *args)

    def subs_dict(Basic self not None, *args):
        warnings.warn("subs_dict() is deprecated. Use subs() instead", DeprecationWarning)
        return self.subs(*args)

    def subs_oldnew(Basic self not None, old, new):
        warnings.warn("subs_oldnew() is deprecated. Use subs() instead", DeprecationWarning)
        return self.subs({old: new})

    def subs(Basic self not None, *args):
        cdef _DictBasic D = get_dict(*args)
        return c2py(symengine.ssubs(self.thisptr, D.c))

    def xreplace(Basic self not None, *args):
        cdef _DictBasic D = get_dict(*args)
        return c2py(symengine.xreplace(self.thisptr, D.c))

    replace = xreplace

    def msubs(Basic self not None, *args):
        cdef _DictBasic D = get_dict(*args)
        return c2py(symengine.msubs(self.thisptr, D.c))

    def as_numer_denom(Basic self not None):
        cdef rcp_const_basic _num, _den
        symengine.as_numer_denom(self.thisptr, symengine.outArg(_num), symengine.outArg(_den))
        return c2py(<rcp_const_basic>_num), c2py(<rcp_const_basic>_den)

    def as_real_imag(Basic self not None):
        cdef rcp_const_basic _real, _imag
        symengine.as_real_imag(self.thisptr, symengine.outArg(_real), symengine.outArg(_imag))
        return c2py(<rcp_const_basic>_real), c2py(<rcp_const_basic>_imag)

    def n(self, unsigned long prec = 53, real=None):
        return evalf(self, prec, real)

    evalf = n

    @property
    def args(self):
        cdef symengine.vec_basic args = deref(self.thisptr).get_args()
        return vec_basic_to_tuple(args)

    @property
    def free_symbols(self):
        cdef symengine.set_basic _set = symengine.free_symbols(deref(self.thisptr))
        return {c2py(<rcp_const_basic>(elem)) for elem in _set}

    @property
    def is_Atom(self):
        return False

    @property
    def is_Symbol(self):
        return False

    @property
    def is_symbol(self):
        return False

    @property
    def is_Dummy(self):
        return False

    @property
    def is_Function(self):
        return False

    @property
    def is_Add(self):
        return False

    @property
    def is_Mul(self):
        return False

    @property
    def is_Pow(self):
        return False

    @property
    def is_Number(self):
        return False

    @property
    def is_number(self):
        return None

    @property
    def is_Float(self):
        return False

    @property
    def is_Rational(self):
        return False

    @property
    def is_Integer(self):
        return False

    @property
    def is_integer(self):
        return False

    @property
    def is_finite(self):
        return None

    @property
    def is_Derivative(self):
        return False

    @property
    def is_AlgebraicNumber(self):
        return False

    @property
    def is_Relational(self):
        return False

    @property
    def is_Equality(self):
        return False

    @property
    def is_Boolean(self):
        return False

    @property
    def is_Not(self):
        return False

    @property
    def is_Matrix(self):
        return False

    @property
    def is_zero(self):
        return is_zero(self)

    @property
    def is_positive(self):
        return is_positive(self)

    @property
    def is_negative(self):
        return is_negative(self)

    @property
    def is_nonpositive(self):
        return is_nonpositive(self)

    @property
    def is_nonnegative(self):
        return is_nonnegative(self)

    @property
    def is_real(self):
        return is_real(self)

    def copy(self):
        return self

    def _symbolic_(self, ring):
        return ring(self._sage_())

    def atoms(self, *types):
        if types:
            s = set()
            if (isinstance(self, types)):
                s.add(self)
            for arg in self.args:
                s.update(arg.atoms(*types))
            return s
        else:
            return self.free_symbols

    def simplify(self, *args, **kwargs):
        return sympify(self._sympy_().simplify(*args, **kwargs))

    def as_coefficients_dict(self):
        d = collections.defaultdict(int)
        d[self] = 1
        return d

    def coeff(self, x, n=1):
        cdef Basic _x = sympify(x)
        require(_x, Symbol)
        cdef Basic _n = sympify(n)
        return c2py(symengine.coeff(deref(self.thisptr), deref(_x.thisptr), deref(_n.thisptr)))

    def has(self, *symbols):
        return any([has_symbol(self, symbol) for symbol in symbols])

    def args_as_sage(Basic self):
        cdef symengine.vec_basic Y = deref(self.thisptr).get_args()
        s = []
        for i in range(Y.size()):
            s.append(c2py(<rcp_const_basic>(Y[i]))._sage_())
        return s

    def args_as_sympy(Basic self):
        cdef symengine.vec_basic Y = deref(self.thisptr).get_args()
        s = []
        for i in range(Y.size()):
            s.append(c2py(<rcp_const_basic>(Y[i]))._sympy_())
        return s

    def __float__(self):
        f = self.n(real=True)
        if not isinstance(f, RealDouble):
            raise TypeError("Can't convert expression to float")
        return float(f)

    def __int__(self):
        return int(float(self))

    def __long__(self):
        return long(float(self))

    def __complex__(self):
        f = self.n(real=False)
        if not isinstance(f, (ComplexDouble, RealDouble)):
            raise TypeError("Can't convert expression to float")
        return complex(f)


def series(ex, x=None, x0=0, n=6, as_deg_coef_pair=False):
    # TODO: check for x0 an infinity, see sympy/core/expr.py
    # TODO: nonzero x0
    # underscored local vars are of symengine.py type
    cdef Basic _ex = sympify(ex)
    syms = _ex.free_symbols
    if not syms:
        return _ex

    cdef Basic _x
    if x is None:
        _x = list(syms)[0]
    else:
        _x = sympify(x)
    require(_x, Symbol)
    if not _x in syms:
        return _ex

    if x0 != 0:
        _ex = _ex.subs({_x: _x + x0})

    cdef RCP[const symengine.Symbol] X = symengine.rcp_static_cast_Symbol(_x.thisptr)
    cdef umap_int_basic umap
    cdef umap_int_basic_iterator iter, iterend

    if not as_deg_coef_pair:
        b = c2py(<symengine.rcp_const_basic>deref(symengine.series(_ex.thisptr, X, n)).as_basic())
        if x0 != 0:
            b = b.subs({_x: _x - x0})
        return b

    umap = deref(symengine.series(_ex.thisptr, X, n)).as_dict()

    iter = umap.begin()
    iterend = umap.end()
    poly = 0
    l = []
    while iter != iterend:
        l.append([deref(iter).first, c2py(<symengine.rcp_const_basic>(deref(iter).second))])
        inc(iter)
    if as_deg_coef_pair:
        return l
    return add(*l)


cdef class Expr(Basic):
    pass


cdef class Symbol(Expr):

    """
    Symbol is a class to store a symbolic variable with a given name.
    """

    def __init__(Basic self, name, *args, **kwargs):
        if type(self) == Symbol:
            self.thisptr = symengine.make_rcp_Symbol(name.encode("utf-8"))
        else:
            self.thisptr = symengine.make_rcp_PySymbol(name.encode("utf-8"), <PyObject*>self)

    def _sympy_(self):
        import sympy
        return sympy.Symbol(str(self))

    def _sage_(self):
        import sage.all as sage
        return sage.SR.symbol(str(self))

    @property
    def name(self):
        return self.__str__()

    @property
    def is_Atom(self):
        return True

    @property
    def is_Symbol(self):
        return True

    @property
    def is_symbol(self):
        return True

    @property
    def is_commutative(self):
        return True

    @property
    def func(self):
        return self.__class__


cdef class Dummy(Symbol):

    def __init__(Basic self, name=None, *args, **kwargs):
        if name is None:
            self.thisptr = symengine.make_rcp_Dummy()
        else:
            self.thisptr = symengine.make_rcp_Dummy(name.encode("utf-8"))

    def _sympy_(self):
        import sympy
        return sympy.Dummy(str(self))

    @property
    def is_Dummy(self):
        return True

    @property
    def func(self):
        return self.__class__


def symarray(prefix, shape, **kwargs):
    """ Creates an nd-array of symbols

    Parameters
    ----------
    prefix: str
    shape: tuple
    \*\*kwargs:
        Passed on to :class:`Symbol`.

    Notes
    -----
    This function requires NumPy.

    """
    arr = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        arr[index] = Symbol('%s_%s' % (prefix, '_'.join(map(str, index))), **kwargs)
    return arr


cdef class Constant(Expr):

    def __cinit__(self, name = None):
        if name is None:
            return
        self.thisptr = symengine.make_rcp_Constant(name.encode("utf-8"))

    @property
    def is_number(self):
        return True

    def _sympy_(self):
        raise Exception("Unknown Constant")

    def _sage_(self):
        raise Exception("Unknown Constant")


cdef class ImaginaryUnit(Complex):

    def __cinit__(Basic self):
        self.thisptr = symengine.I

I = ImaginaryUnit()


cdef class Pi(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.pi

    def _sympy_(self):
        import sympy
        return sympy.pi

    def _sage_(self):
        import sage.all as sage
        return sage.pi

pi = Pi()


cdef class Exp1(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.E

    def _sympy_(self):
        import sympy
        return sympy.E

    def _sage_(self):
        import sage.all as sage
        return sage.e

E = Exp1()


cdef class GoldenRatio(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.GoldenRatio

    def _sympy_(self):
        import sympy
        return sympy.GoldenRatio

    def _sage_(self):
        import sage.all as sage
        return sage.golden_ratio

golden_ratio = GoldenRatio()


cdef class Catalan(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.Catalan

    def _sympy_(self):
        import sympy
        return sympy.Catalan

    def _sage_(self):
        import sage.all as sage
        return sage.catalan

catalan = Catalan()


cdef class EulerGamma(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.EulerGamma

    def _sympy_(self):
        import sympy
        return sympy.EulerGamma

    def _sage_(self):
        import sage.all as sage
        return sage.euler_gamma

eulergamma = EulerGamma()


cdef class Boolean(Expr):

    def logical_not(self):
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Boolean(self.thisptr)).logical_not()))


cdef class BooleanAtom(Boolean):

    @property
    def is_Boolean(self):
        return True

    @property
    def is_Atom(self):
        return True


cdef class BooleanTrue(BooleanAtom):

    def __cinit__(Basic self):
        self.thisptr = symengine.boolTrue

    def _sympy_(self):
        import sympy
        return sympy.S.true

    def _sage_(self):
        return True

    def __nonzero__(self):
        return True

    def __bool__(self):
        return True


true = BooleanTrue()


cdef class BooleanFalse(BooleanAtom):

    def __cinit__(Basic self):
        self.thisptr = symengine.boolFalse

    def _sympy_(self):
        import sympy
        return sympy.S.false

    def _sage_(self):
        return False

    def __nonzero__(self):
        return False

    def __bool__(self):
        return False

false = BooleanFalse()


class And(Boolean):

    def __new__(cls, *args):
        return logical_and(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.And(*s)


class Or(Boolean):

    def __new__(cls, *args):
        return logical_or(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Or(*s)


class Not(Boolean):

    def __new__(cls, x):
        return logical_not(x)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()[0]
        return sympy.Not(s)


class Xor(Boolean):

    def __new__(cls, *args):
        return logical_xor(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Xor(*s)


class Relational(Boolean):

    @property
    def is_Relational(self):
        return True

Rel = Relational


class Equality(Relational):

    def __new__(cls, *args):
        return eq(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Equality(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.eq(*s)

    @property
    def is_Equality(self):
        return True

    func = __class__


Eq = Equality


class Unequality(Relational):

    def __new__(cls, *args):
        return ne(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Unequality(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.ne(*s)

    func = __class__


Ne = Unequality


class LessThan(Relational):

    def __new__(cls, *args):
        return le(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.LessThan(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.le(*s)


Le = LessThan


class StrictLessThan(Relational):

    def __new__(cls, *args):
        return lt(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.StrictLessThan(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.lt(*s)


Lt = StrictLessThan


cdef class Number(Expr):
    @property
    def is_Atom(self):
        return True

    @property
    def is_Number(self):
        return True

    @property
    def is_number(self):
        return True

    @property
    def is_commutative(self):
        return True

    @property
    def is_positive(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_positive()

    @property
    def is_negative(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_negative()

    @property
    def is_nonzero(self):
        return not (self.is_complex or self.is_zero)

    @property
    def is_nonnegative(self):
        return not (self.is_complex or self.is_negative)

    @property
    def is_nonpositive(self):
        return not (self.is_complex or self.is_positive)

    @property
    def is_complex(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_complex()

    @property
    def real(self):
        return self

    @property
    def imag(self):
        return S.Zero


class Rational(Number):

    def __new__(cls, p, q):
        return Integer(p)/q

    @property
    def is_Rational(self):
        return True

    @property
    def is_rational(self):
        return True

    @property
    def is_real(self):
        return True

    @property
    def is_finite(self):
        return True

    @property
    def is_integer(self):
        return False

    @property
    def p(self):
        return self.get_num_den()[0]

    @property
    def q(self):
        return self.get_num_den()[1]

    def get_num_den(Basic self):
        cdef RCP[const symengine.Integer] _num, _den
        symengine.get_num_den(deref(symengine.rcp_static_cast_Rational(self.thisptr)),
                           symengine.outArg_Integer(_num), symengine.outArg_Integer(_den))
        return [c2py(<rcp_const_basic>_num), c2py(<rcp_const_basic>_den)]

    def _sympy_(self):
        rat = self.get_num_den()
        return rat[0]._sympy_() / rat[1]._sympy_()

    def _sage_(self):
        try:
            from sage.symbolic.symengine_conversions import convert_to_rational
            return convert_to_rational(self)
        except ImportError:
            rat = self.get_num_den()
            return rat[0]._sage_() / rat[1]._sage_()

    @property
    def func(self):
        return self.__class__


class Integer(Rational):

    def __new__(cls, i):
        i = int(i)
        cdef int i_
        cdef symengine.integer_class i__
        cdef string tmp
        try:
            # Try to convert "i" to int
            i_ = i
            int_ok = True
        except OverflowError:
            # Too big, need to use mpz
            int_ok = False
            tmp = str(i).encode("utf-8")
            i__ = symengine.integer_class(tmp)
        # Note: all other exceptions are left intact
        if int_ok:
            return c2py(<rcp_const_basic>symengine.integer(i_))
        else:
            return c2py(<rcp_const_basic>symengine.integer(i__))

    @property
    def is_Integer(self):
        return True

    @property
    def is_integer(self):
        return True

    def doit(self, **hints):
        return self

    def __hash__(Basic self):
        return deref(self.thisptr).hash()

    def __richcmp__(a, b, int op):
        A = _sympify(a, False)
        B = _sympify(b, False)
        if not (isinstance(A, Integer) and isinstance(B, Integer)):
            if (op == 2):
                return False
            elif (op == 3):
                return True
            return NotImplemented
        return Integer._richcmp_(A, B, op)

    def _richcmp_(Basic A, Basic B, int op):
        cdef int i = deref(symengine.rcp_static_cast_Integer(A.thisptr)).compare(deref(symengine.rcp_static_cast_Integer(B.thisptr)))
        if (op == 0):
            return i < 0
        elif (op == 1):
            return i <= 0
        elif (op == 2):
            return i == 0
        elif (op == 3):
            return i != 0
        elif (op == 4):
            return i > 0
        elif (op == 5):
            return i >= 0
        else:
            return NotImplemented

    def __floordiv__(x, y):
        return quotient(x, y)

    def __mod__(x, y):
        return mod(x, y)

    def __divmod__(x, y):
        return quotient_mod(x, y)

    def _sympy_(Basic self):
        import sympy
        return sympy.Integer(deref(self.thisptr).__str__().decode("utf-8"))

    def _sage_(Basic self):
        try:
            from sage.symbolic.symengine_conversions import convert_to_integer
            return convert_to_integer(self)
        except ImportError:
            import sage.all as sage
            return sage.Integer(str(self))

    def __int__(Basic self):
        return int(str(self))

    @property
    def p(self):
        return int(self)

    @property
    def q(self):
        return 1

    def get_num_den(Basic self):
        return self, 1

    @property
    def func(self):
        return self.__class__


def dps_to_prec(n):
    """Return the number of bits required to represent n decimals accurately."""
    return max(1, int(round((int(n)+1)*3.3219280948873626)))


class BasicMeta(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, self._classes)

class Float(Number):

    @property
    def is_rational(self):
        return None

    @property
    def is_irrational(self):
        return None

    @property
    def is_real(self):
        return True

    @property
    def is_Float(self):
        return True

    def __new__(cls, num, dps=None, precision=None):
        if cls is not Float:
            return super(Float, cls).__new__(cls)

        if dps is not None and precision is not None:
            raise ValueError('Both decimal and binary precision supplied. '
                             'Supply only one. ')
        if dps is None and precision is None:
            dps = 15
        if precision is None:
            precision = dps_to_prec(dps)

        IF HAVE_SYMENGINE_MPFR:
            if precision > 53:
                if isinstance(num, RealMPFR) and precision == num.get_prec():
                    return num
                return RealMPFR(str(num), precision)
        if precision > 53:
            raise ValueError('RealMPFR unavailable for high precision numerical values.')
        elif isinstance(num, RealDouble):
            return num
        else:
            return RealDouble(float(num))


RealNumber = Float


class RealDouble(Float):

    def __new__(cls, i):
        cdef double i_ = i
        return c2py(symengine.make_rcp_RealDouble(i_))

    def _sympy_(Basic self):
        import sympy
        return sympy.Float(deref(self.thisptr).__str__().decode("utf-8"))

    def _sage_(Basic self):
        import sage.all as sage
        return sage.RealDoubleField()(float(self))

    def __float__(Basic self):
        return deref(symengine.rcp_static_cast_RealDouble(self.thisptr)).as_double()

    def __complex__(self):
        return complex(float(self))


cdef class ComplexBase(Number):

    def real_part(Basic self):
        return c2py(<rcp_const_basic>deref(symengine.rcp_static_cast_ComplexBase(self.thisptr)).real_part())

    def imaginary_part(Basic self):
        return c2py(<rcp_const_basic>deref(symengine.rcp_static_cast_ComplexBase(self.thisptr)).imaginary_part())

    @property
    def real(self):
        return self.real_part()

    @property
    def imag(self):
        return self.imaginary_part()


cdef class ComplexDouble(ComplexBase):

    def __cinit__(self, i = None):
        if i is None:
            return
        cdef double complex i_ = i
        self.thisptr = symengine.make_rcp_ComplexDouble(i_)

    def _sympy_(self):
        import sympy
        return self.real_part()._sympy_() + sympy.I * self.imaginary_part()._sympy_()

    def _sage_(self):
        import sage.all as sage
        return self.real_part()._sage_() + sage.I * self.imaginary_part()._sage_()

    def __complex__(Basic self):
        return deref(symengine.rcp_static_cast_ComplexDouble(self.thisptr)).as_complex_double()


class RealMPFR(Float):

    IF HAVE_SYMENGINE_MPFR:
        def __new__(cls, i = None, long prec = 53, unsigned base = 10):
            if i is None:
                return
            cdef string i_ = str(i).encode("utf-8")
            cdef symengine.mpfr_class m
            m = symengine.mpfr_class(i_, prec, base)
            return c2py(<rcp_const_basic>symengine.real_mpfr(symengine.std_move_mpfr(m)))

        def get_prec(Basic self):
            return Integer(deref(symengine.rcp_static_cast_RealMPFR(self.thisptr)).get_prec())

        def _sympy_(self):
            import sympy
            cdef long prec_ = self.get_prec()
            prec = max(1, int(round(prec_/3.3219280948873626)-1))
            return sympy.Float(str(self), prec)

        def _sage_(self):
            try:
                from sage.symbolic.symengine_conversions import convert_to_real_number
                return convert_to_real_number(self)
            except ImportError:
                import sage.all as sage
                return sage.RealField(int(self.get_prec()))(str(self))
    ELSE:
        pass


cdef class ComplexMPC(ComplexBase):
    IF HAVE_SYMENGINE_MPC:
        def __cinit__(self, i = None, j = 0, long prec = 53, unsigned base = 10):
            if i is None:
                return
            cdef string i_ = ("(" + str(i) + " " + str(j) + ")").encode("utf-8")
            cdef symengine.mpc_class m = symengine.mpc_class(i_, prec, base)
            self.thisptr = <rcp_const_basic>symengine.complex_mpc(symengine.std_move_mpc(m))

        def _sympy_(self):
            import sympy
            return self.real_part()._sympy_() + sympy.I * self.imaginary_part()._sympy_()

        def _sage_(self):
            try:
                from sage.symbolic.symengine_conversions import convert_to_mpcomplex_number
                return convert_to_mpcomplex_number(self)
            except ImportError:
                import sage.all as sage
                return sage.MPComplexField(int(self.get_prec()))(str(self.real_part()), str(self.imaginary_part()))
    ELSE:
        pass


cdef class Complex(ComplexBase):

    def _sympy_(self):
        import sympy
        return self.real_part()._sympy_() + sympy.I * self.imaginary_part()._sympy_()

    def _sage_(self):
        import sage.all as sage
        return self.real_part()._sage_() + sage.I * self.imaginary_part()._sage_()


cdef class Infinity(Number):

    @property
    def is_infinite(self):
        return True

    def __cinit__(Basic self):
        self.thisptr = symengine.Inf

    def _sympy_(self):
        import sympy
        return sympy.oo

    def _sage_(self):
        import sage.all as sage
        return sage.oo

oo = Infinity()


cdef class NegativeInfinity(Number):

    @property
    def is_infinite(self):
        return True

    def __cinit__(Basic self):
        self.thisptr = symengine.neg(symengine.Inf)

    def _sympy_(self):
        import sympy
        return -sympy.oo

    def _sage_(self):
        import sage.all as sage
        return -sage.oo

minus_oo = NegativeInfinity()


cdef class ComplexInfinity(Number):

    @property
    def is_infinite(self):
        return True

    def __cinit__(Basic self):
        self.thisptr = symengine.ComplexInf

    def _sympy_(self):
        import sympy
        return sympy.zoo

    def _sage_(self):
        import sage.all as sage
        return sage.unsigned_infinity

zoo = ComplexInfinity()


cdef class NaN(Number):

    @property
    def is_rational(self):
        return None

    @property
    def is_integer(self):
        return None

    @property
    def is_real(self):
        return None

    @property
    def is_finite(self):
        return None

    def __cinit__(Basic self):
        self.thisptr = symengine.Nan

    def _sympy_(self):
        import sympy
        return sympy.nan

    def _sage_(self):
        import sage.all as sage
        return sage.NaN

nan = NaN()


class Zero(Integer):
    def __new__(cls):
        cdef Basic r = Number.__new__(Zero)
        r.thisptr = <rcp_const_basic>symengine.integer(0)
        return r

zero = Zero()


class One(Integer):
    def __new__(cls):
        cdef Basic r = Number.__new__(One)
        r.thisptr = <rcp_const_basic>symengine.integer(1)
        return r

one = One()


class NegativeOne(Integer):
    def __new__(cls):
        cdef Basic r = Number.__new__(NegativeOne)
        r.thisptr = <rcp_const_basic>symengine.integer(-1)
        return r

minus_one = NegativeOne()


class Half(Rational):
    def __new__(cls):
        cdef Basic q = Number.__new__(Half)
        q.thisptr = <rcp_const_basic>symengine.rational(1, 2)
        return q

half = Half()


class AssocOp(Expr):

    @classmethod
    def make_args(cls, expr):
        if isinstance(expr, cls):
            return expr.args
        else:
            return (sympify(expr),)


class Add(AssocOp):

    identity = 0

    def __new__(cls, *args, **kwargs):
        cdef symengine.vec_basic v_ = iter_to_vec_basic(args)
        return c2py(symengine.add(v_))

    @classmethod
    def _from_args(self, args):
        if len(args) == 0:
            return self.identity
        elif len(args) == 1:
            return args[0]

        return Add(*args)

    @property
    def is_Add(self):
        return True

    def _sympy_(self):
        from sympy import Add
        return Add(*self.args)

    def _sage_(Basic self):
        cdef RCP[const symengine.Add] X = symengine.rcp_static_cast_Add(self.thisptr)
        cdef rcp_const_basic a, b
        deref(X).as_two_terms(symengine.outArg(a), symengine.outArg(b))
        return c2py(a)._sage_() + c2py(b)._sage_()

    @property
    def func(self):
        return self.__class__

    def as_coefficients_dict(Basic self):
        cdef RCP[const symengine.Add] X = symengine.rcp_static_cast_Add(self.thisptr)
        cdef umap_basic_num umap
        cdef umap_basic_num_iterator iter, iterend
        d = collections.defaultdict(int)
        d[Integer(1)] = c2py(<rcp_const_basic>(deref(X).get_coef()))
        umap = deref(X).get_dict()
        iter = umap.begin()
        iterend = umap.end()
        while iter != iterend:
            d[c2py(<rcp_const_basic>(deref(iter).first))] =\
                    c2py(<rcp_const_basic>(deref(iter).second))
            inc(iter)
        return d


class Mul(AssocOp):

    identity = 1

    def __new__(cls, *args, **kwargs):
        cdef symengine.vec_basic v_ = iter_to_vec_basic(args)
        return c2py(symengine.mul(v_))

    @classmethod
    def _from_args(self, args):
        if len(args) == 0:
            return self.identity
        elif len(args) == 1:
            return args[0]

        return Mul(*args)

    @property
    def is_Mul(self):
        return True

    def _sympy_(self):
        from sympy import Mul
        return Mul(*self.args)

    def _sage_(Basic self):
        cdef RCP[const symengine.Mul] X = symengine.rcp_static_cast_Mul(self.thisptr)
        cdef rcp_const_basic a, b
        deref(X).as_two_terms(symengine.outArg(a), symengine.outArg(b))
        return c2py(a)._sage_() * c2py(b)._sage_()

    @property
    def func(self):
        return self.__class__

    def as_coefficients_dict(Basic self):
        cdef RCP[const symengine.Mul] X = symengine.rcp_static_cast_Mul(self.thisptr)
        cdef RCP[const symengine.Integer] one = symengine.integer(1)
        cdef map_basic_basic dict = deref(X).get_dict()
        d = collections.defaultdict(int)
        d[c2py(<rcp_const_basic>symengine.mul_from_dict(\
                <RCP[const symengine.Number]>(one),
                symengine.std_move_map_basic_basic(dict)))] =\
                c2py(<rcp_const_basic>deref(X).get_coef())
        return d


class Pow(Expr):

    def __new__(cls, a, b):
        return _sympify(a) ** b

    @property
    def base(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        return c2py(deref(X).get_base())

    @property
    def exp(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        return c2py(deref(X).get_exp())

    def as_base_exp(self):
        return self.base, self.exp

    @property
    def is_Pow(self):
        return True

    @property
    def is_commutative(self):
        return (self.base.is_commutative and self.exp.is_commutative)

    def _sympy_(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        base = c2py(deref(X).get_base())
        exp = c2py(deref(X).get_exp())
        return base._sympy_() ** exp._sympy_()

    def _sage_(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        base = c2py(deref(X).get_base())
        exp = c2py(deref(X).get_exp())
        return base._sage_() ** exp._sage_()

    @property
    def func(self):
        return self.__class__


class Function(Expr):

    def __new__(cls, *args, **kwargs):
        if cls == Function and len(args) == 1:
            return UndefFunction(args[0])
        return super(Function, cls).__new__(cls)

    @property
    def is_Function(self):
        return True

    def func(self, *values):
        import sys
        return getattr(sys.modules[__name__], self.__class__.__name__.lower())(*values)


class OneArgFunction(Function):

    def get_arg(Basic self):
        cdef RCP[const symengine.OneArgFunction] X = symengine.rcp_static_cast_OneArgFunction(self.thisptr)
        return c2py(deref(X).get_arg())

    def _sympy_(self):
        import sympy
        return getattr(sympy, self.__class__.__name__)(self.get_arg()._sympy_())

    def _sage_(self):
        import sage.all as sage
        return getattr(sage, self.__class__.__name__.lower())(self.get_arg()._sage_())


class HyperbolicFunction(OneArgFunction):
    pass

class TrigFunction(OneArgFunction):
    pass

class gamma(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.gamma(X.thisptr))

class LambertW(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.lambertw(X.thisptr))

    def _sage_(self):
        import sage.all as sage
        return sage.lambert_w(self.get_arg()._sage_())

class zeta(Function):
    def __new__(cls, s, a = None):
        cdef Basic S = sympify(s)
        if a == None:
            return c2py(symengine.zeta(S.thisptr))
        cdef Basic A = sympify(a)
        return c2py(symengine.zeta(S.thisptr, A.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.zeta(*self.args_as_sympy())

class dirichlet_eta(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.dirichlet_eta(X.thisptr))

class KroneckerDelta(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.kronecker_delta(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.KroneckerDelta(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        return sage.kronecker_delta(*self.args_as_sage())

class LeviCivita(Function):
    def __new__(cls, *args):
        cdef symengine.vec_basic v = iter_to_vec_basic(args)
        return c2py(symengine.levi_civita(v))

    def _sympy_(self):
        import sympy
        return sympy.LeviCivita(*self.args_as_sympy())

class erf(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.erf(X.thisptr))

class erfc(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.erfc(X.thisptr))

class lowergamma(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.lowergamma(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.lowergamma(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        return sage.gamma_inc_lower(*self.args_as_sage())

class uppergamma(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.uppergamma(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.uppergamma(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        return sage.gamma_inc(*self.args_as_sage())

class loggamma(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.loggamma(X.thisptr))

    def _sage_(self):
        import sage.all as sage
        return sage.log_gamma(self.get_arg()._sage_())

class beta(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.beta(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.beta(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        return sage.beta(*self.args_as_sage())

class polygamma(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.polygamma(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.polygamma(*self.args_as_sympy())

class sign(OneArgFunction):

    @property
    def is_complex(self):
        return True

    @property
    def is_finite(self):
        return True

    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sign(X.thisptr))

class floor(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.floor(X.thisptr))

class ceiling(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.ceiling(X.thisptr))

    def _sage_(self):
        import sage.all as sage
        return sage.ceil(self.get_arg()._sage_())

class conjugate(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.conjugate(X.thisptr))

class log(OneArgFunction):
    def __new__(cls, x, y=None):
        cdef Basic X = sympify(x)
        if y == None:
            return c2py(symengine.log(X.thisptr))
        cdef Basic Y = sympify(y)
        return c2py(symengine.log(X.thisptr, Y.thisptr))

class sin(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sin(X.thisptr))

class cos(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.cos(X.thisptr))

class tan(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.tan(X.thisptr))

class cot(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.cot(X.thisptr))

class sec(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sec(X.thisptr))

class csc(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.csc(X.thisptr))

class asin(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asin(X.thisptr))

class acos(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acos(X.thisptr))

class atan(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.atan(X.thisptr))

class acot(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acot(X.thisptr))

class asec(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asec(X.thisptr))

class acsc(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acsc(X.thisptr))

class sinh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sinh(X.thisptr))

class cosh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.cosh(X.thisptr))

class tanh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.tanh(X.thisptr))

class coth(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.coth(X.thisptr))

class sech(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sech(X.thisptr))

class csch(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.csch(X.thisptr))

class asinh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asinh(X.thisptr))

class acosh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acosh(X.thisptr))

class atanh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.atanh(X.thisptr))

class acoth(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acoth(X.thisptr))

class asech(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asech(X.thisptr))

class acsch(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acsch(X.thisptr))

class atan2(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.atan2(X.thisptr, Y.thisptr))

# For backwards compatibility

Sin = sin
Cos = cos
Tan = tan
Cot = cot
Sec = sec
Csc = csc
ASin = asin
ACos = acos
ATan = atan
ACot = acot
ASec = asec
ACsc = acsc
Sinh = sinh
Cosh = cosh
Tanh = tanh
Coth = coth
Sech = sech
Csch = csch
ASinh = asinh
ACosh = acosh
ATanh = atanh
ACoth = acoth
ASech = asech
ACsch = acsch
ATan2 = atan2
Log = log
Gamma = gamma

add = Add
mul = Mul


class UnevaluatedExpr(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.unevaluated_expr(X.thisptr))

    @property
    def is_number(self):
        return self.args[0].is_number

    @property
    def is_integer(self):
        return self.args[0].is_integer

    @property
    def is_finite(self):
        return self.args[0].is_finite


class Abs(OneArgFunction):

    @property
    def is_real(self):
        return True

    @property
    def is_negative(self):
        return False

    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.abs(X.thisptr))

    def _sympy_(Basic self):
        cdef RCP[const symengine.Abs] X = symengine.rcp_static_cast_Abs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sympy_()
        return abs(arg)

    def _sage_(Basic self):
        cdef RCP[const symengine.Abs] X = symengine.rcp_static_cast_Abs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sage_()
        return abs(arg)

    @property
    def func(self):
        return self.__class__

class FunctionSymbol(Function):

    def get_name(Basic self):
        cdef RCP[const symengine.FunctionSymbol] X = \
            symengine.rcp_static_cast_FunctionSymbol(self.thisptr)
        name = deref(X).get_name().decode("utf-8")
        return str(name)

    def _sympy_(self):
        import sympy
        name = self.get_name()
        return sympy.Function(name)(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        name = self.get_name()
        return sage.function(name, *self.args_as_sage())

    def func(self, *values):
        name = self.get_name()
        return function_symbol(name, *values)


class UndefFunction(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, *values):
        return function_symbol(self.name, *values)


cdef rcp_const_basic pynumber_to_symengine(PyObject* o1):
    cdef Basic X = sympify(<object>o1)
    return X.thisptr

cdef PyObject* symengine_to_sage(rcp_const_basic o1):
    import sage.all as sage
    t = sage.SR(c2py(o1)._sage_())
    Py_XINCREF(<PyObject*>t)
    return <PyObject*>(t)

cdef PyObject* symengine_to_sympy(rcp_const_basic o1):
    t = c2py(o1)._sympy_()
    Py_XINCREF(<PyObject*>t)
    return <PyObject*>(t)

cdef RCP[const symengine.Number] sympy_eval(PyObject* o1, long bits):
    prec = max(1, int(round(bits/3.3219280948873626)-1))
    cdef Number X = sympify((<object>o1).n(prec))
    return symengine.rcp_static_cast_Number(X.thisptr)

cdef RCP[const symengine.Number] sage_eval(PyObject* o1, long bits):
    cdef Number X = sympify((<object>o1).n(bits))
    return symengine.rcp_static_cast_Number(X.thisptr)

cdef rcp_const_basic sage_diff(PyObject* o1, rcp_const_basic symbol):
    cdef Basic X = sympify((<object>o1).diff(c2py(symbol)._sage_()))
    return X.thisptr

cdef rcp_const_basic sympy_diff(PyObject* o1, rcp_const_basic symbol):
    cdef Basic X = sympify((<object>o1).diff(c2py(symbol)._sympy_()))
    return X.thisptr

def create_sympy_module():
    cdef PyModule s = PyModule.__new__(PyModule)
    s.thisptr = symengine.make_rcp_PyModule(&symengine_to_sympy, &pynumber_to_symengine, &sympy_eval,
                                    &sympy_diff)
    return s

def create_sage_module():
    cdef PyModule s = PyModule.__new__(PyModule)
    s.thisptr = symengine.make_rcp_PyModule(&symengine_to_sage, &pynumber_to_symengine, &sage_eval,
                                    &sage_diff)
    return s

sympy_module = create_sympy_module()
sage_module = create_sage_module()

cdef class PyNumber(Number):
    def __cinit__(self, obj = None, PyModule module = None):
        if obj is None:
            return
        Py_XINCREF(<PyObject*>(obj))
        self.thisptr = symengine.make_rcp_PyNumber(<PyObject*>(obj), module.thisptr)

    def _sympy_(self):
        import sympy
        return sympy.sympify(self.pyobject())

    def _sage_(self):
        import sage.all as sage
        return sage.SR(self.pyobject())

    def pyobject(self):
        return <object>deref(symengine.rcp_static_cast_PyNumber(self.thisptr)).get_py_object()


class PyFunction(FunctionSymbol):

    def __init__(Basic self, pyfunction = None, args = None, pyfunction_class=None, module=None):
        if pyfunction is None:
            return
        cdef symengine.vec_basic v = iter_to_vec_basic(args)
        cdef PyFunctionClass _pyfunction_class = get_function_class(pyfunction_class, module)
        cdef PyObject* _pyfunction = <PyObject*>pyfunction
        Py_XINCREF(_pyfunction)
        self.thisptr = symengine.make_rcp_PyFunction(v, _pyfunction_class.thisptr, _pyfunction)

    def _sympy_(self):
        import sympy
        return sympy.sympify(self.pyobject())

    def _sage_(self):
        import sage.all as sage
        return sage.SR(self.pyobject())

    def pyobject(Basic self):
        return <object>deref(symengine.rcp_static_cast_PyFunction(self.thisptr)).get_py_object()

cdef class PyFunctionClass(object):

    def __cinit__(self, function, PyModule module not None):
        self.thisptr = symengine.make_rcp_PyFunctionClass(<PyObject*>(function), str(function).encode("utf-8"),
                                module.thisptr)

# TODO: remove this once SymEngine conversions are available in Sage.
def wrap_sage_function(func):
    return PyFunction(func, func.operands(), func.operator(), sage_module)


class Max(Function):

    def __new__(cls, *args):
        return _max(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Max(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.max(*s)

    @property
    def func(self):
        return self.__class__

class Min(Function):

    def __new__(cls, *args):
        return _min(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Min(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.min(*s)

    @property
    def func(self):
        return self.__class__


class Derivative(Expr):

    def __new__(self, expr, *variables):
        if len(variables) == 1 and is_sequence(variables[0]):
            return diff(expr, *variables[0])
        return diff(expr, *variables)

    @property
    def is_Derivative(self):
        return True

    @property
    def expr(Basic self):
        cdef RCP[const symengine.Derivative] X = symengine.rcp_static_cast_Derivative(self.thisptr)
        return c2py(deref(X).get_arg())

    @property
    def variables(self):
        return self.args[1:]

    def _sympy_(Basic self):
        cdef RCP[const symengine.Derivative] X = \
            symengine.rcp_static_cast_Derivative(self.thisptr)
        arg = c2py(deref(X).get_arg())._sympy_()
        cdef symengine.multiset_basic Y = deref(X).get_symbols()
        s = []
        for i in Y:
            s.append(c2py(<rcp_const_basic>(i))._sympy_())
        import sympy
        return sympy.Derivative(arg, *s)

    def _sage_(Basic self):
        cdef RCP[const symengine.Derivative] X = \
            symengine.rcp_static_cast_Derivative(self.thisptr)
        arg = c2py(deref(X).get_arg())._sage_()
        cdef symengine.multiset_basic Y = deref(X).get_symbols()
        s = []
        for i in Y:
            s.append(c2py(<rcp_const_basic>(i))._sage_())
        return arg.diff(*s)

    @property
    def func(self):
        return self.__class__


class Subs(Expr):

    def __new__(self, expr, variables, point):
        return sympify(expr).subs(variables, point)

    @property
    def expr(Basic self):
        cdef RCP[const symengine.Subs] me = symengine.rcp_static_cast_Subs(self.thisptr)
        return c2py(deref(me).get_arg())

    @property
    def variables(Basic self):
        cdef RCP[const symengine.Subs] me = symengine.rcp_static_cast_Subs(self.thisptr)
        cdef symengine.vec_basic variables = deref(me).get_variables()
        return vec_basic_to_tuple(variables)

    @property
    def point(Basic self):
        cdef RCP[const symengine.Subs] me = symengine.rcp_static_cast_Subs(self.thisptr)
        cdef symengine.vec_basic point = deref(me).get_point()
        return vec_basic_to_tuple(point)

    def _sympy_(Basic self):
        cdef RCP[const symengine.Subs] X = symengine.rcp_static_cast_Subs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sympy_()
        cdef symengine.vec_basic V = deref(X).get_variables()
        cdef symengine.vec_basic P = deref(X).get_point()
        v = []
        p = []
        for i in range(V.size()):
            v.append(c2py(<rcp_const_basic>(V[i]))._sympy_())
            p.append(c2py(<rcp_const_basic>(P[i]))._sympy_())
        import sympy
        return sympy.Subs(arg, v, p)

    def _sage_(Basic self):
        cdef RCP[const symengine.Subs] X = symengine.rcp_static_cast_Subs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sage_()
        cdef symengine.vec_basic V = deref(X).get_variables()
        cdef symengine.vec_basic P = deref(X).get_point()
        v = {}
        for i in range(V.size()):
            v[c2py(<rcp_const_basic>(V[i]))._sage_()] = \
                c2py(<rcp_const_basic>(P[i]))._sage_()
        return arg.subs(v)

    @property
    def func(self):
        return self.__class__


class Piecewise(Function):

    def __new__(self, *args):
        return piecewise(*args)

    def _sympy_(self):
        import sympy
        a = self.args
        l = []
        for i in range(0, len(a), 2):
            l.append((a[i]._sympy_(), a[i + 1]._sympy_()))
        return sympy.Piecewise(*l)


cdef class Set(Expr):

    def intersection(self, a):
        cdef Set other = sympify(a)
        cdef RCP[const symengine.Set] other_ = symengine.rcp_static_cast_Set(other.thisptr)
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Set(self.thisptr))
                    .set_intersection(other_)))

    def union(self, a):
        cdef Set other = sympify(a)
        cdef RCP[const symengine.Set] other_ = symengine.rcp_static_cast_Set(other.thisptr)
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Set(self.thisptr))
                    .set_union(other_)))

    def complement(self, a):
        cdef Set other = sympify(a)
        cdef RCP[const symengine.Set] other_ = symengine.rcp_static_cast_Set(other.thisptr)
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Set(self.thisptr))
                    .set_complement(other_)))

    def contains(self, a):
        cdef Basic a_ = sympify(a)
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Set(self.thisptr))
                    .contains(a_.thisptr)))


class Interval(Set):

    def __new__(self, *args):
        return interval(*args)

    def _sympy_(self):
        import sympy
        return sympy.Interval(*[arg._sympy_() for arg in self.args])


class EmptySet(Set):

    def __new__(self):
        return emptyset()

    def _sympy_(self):
        import sympy
        return sympy.S.EmptySet

    @property
    def func(self):
        return self.__class__


class Reals(Set):

    def __new__(self):
        return reals()

    def _sympy_(self):
        import sympy
        return sympy.S.Reals

    @property
    def func(self):
        return self.__class__


class Integers(Set):

    def __new__(self):
        return integers()

    def _sympy_(self):
        import sympy
        return sympy.S.Integers

    @property
    def func(self):
        return self.__class__


class UniversalSet(Set):

    def __new__(self):
        return universalset()

    def _sympy_(self):
        import sympy
        return sympy.S.UniversalSet

    @property
    def func(self):
        return self.__class__


class FiniteSet(Set):

    def __new__(self, *args):
        return finiteset(*args)

    def _sympy_(self):
        import sympy
        return sympy.FiniteSet(*[arg._sympy_() for arg in self.args])


class Contains(Boolean):

    def __new__(self, expr, sset):
        return contains(expr, sset)

    def _sympy_(self):
        import sympy
        return sympy.Contains(*[arg._sympy_() for arg in self.args])


class Union(Set):

    def __new__(self, *args):
        return set_union(*args)

    def _sympy_(self):
        import sympy
        return sympy.Union(*[arg._sympy_() for arg in self.args])


class Complement(Set):

    def __new__(self, universe, container):
        return set_complement(universe, container)

    def _sympy_(self):
        import sympy
        return sympy.Complement(*[arg._sympy_() for arg in self.args])


class ConditionSet(Set):

    def __new__(self, sym, condition):
        return conditionset(sym, condition)


class ImageSet(Set):

    def __new__(self, sym, expr, base):
        return imageset(sym, expr, base)



cdef class MatrixBase:

    @property
    def is_Matrix(self):
        return True

    def __richcmp__(a, b, int op):
        A = _sympify(a, False)
        B = _sympify(b, False)
        if not (isinstance(A, MatrixBase) and isinstance(B, MatrixBase)):
            if (op == 2):
                return False
            elif (op == 3):
                return True
            return NotImplemented
        return A._richcmp_(B, op)

    def _richcmp_(MatrixBase A, MatrixBase B, int op):
        if (op == 2):
            return deref(A.thisptr).eq(deref(B.thisptr))
        elif (op == 3):
            return not deref(A.thisptr).eq(deref(B.thisptr))
        else:
            return NotImplemented

    def __dealloc__(self):
        del self.thisptr

    def _symbolic_(self, ring):
        return ring(self._sage_())

    # TODO: fix this
    def __hash__(self):
        return 0


class MatrixError(Exception):
    pass


class ShapeError(ValueError, MatrixError):
    """Wrong matrix shape"""
    pass


class NonSquareMatrixError(ShapeError):
    pass

cdef class DenseMatrixBase(MatrixBase):
    """
    Represents a two-dimensional dense matrix.

    Examples
    ========

    Empty matrix:

    >>> DenseMatrix(3, 2)

    2D Matrix:

    >>> DenseMatrix(3, 2, [1, 2, 3, 4, 5, 6])
    [1, 2]
    [3, 4]
    [5, 6]

    >>> DenseMatrix([[1, 2], [3, 4], [5, 6]])
    [1, 2]
    [3, 4]
    [5, 6]

    """

    def __cinit__(self, row=None, col=None, v=None):
        if row is None:
            self.thisptr = new symengine.DenseMatrix(0, 0)
            return
        if v is None and col is not None:
            self.thisptr = new symengine.DenseMatrix(row, col)
            return
        if col is None:
            v = row
            row = 0
        cdef symengine.vec_basic v_
        cdef DenseMatrixBase A
        cdef Basic e_
        #TODO: Add a constructor to DenseMatrix in C++
        if (isinstance(v, DenseMatrixBase)):
            matrix_to_vec(v, v_)
            if col is None:
                row = v.nrows()
                col = v.ncols()
            self.thisptr = new symengine.DenseMatrix(row, col, v_)
            return
        for e in v:
            f = sympify(e)
            if isinstance(f, DenseMatrixBase):
                matrix_to_vec(f, v_)
                if col is None:
                    row = row + f.nrows()
                continue
            try:
                for e_ in f:
                    v_.push_back(e_.thisptr)
                if col is None:
                    row = row + 1
            except TypeError:
                e_ = f
                v_.push_back(e_.thisptr)
                if col is None:
                    row = row + 1
        if (row == 0):
            if (v_.size() != 0):
                self.thisptr = new symengine.DenseMatrix(0, 0, v_)
                raise ValueError("sizes don't match.")
            else:
                self.thisptr = new symengine.DenseMatrix(0, 0, v_)
        else:
            self.thisptr = new symengine.DenseMatrix(row, v_.size() / row, v_)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return deref(self.thisptr).__str__().decode("utf-8")

    def __add__(a, b):
        a = _sympify(a, False)
        b = _sympify(b, False)
        if not isinstance(a, MatrixBase) or not isinstance(b, MatrixBase):
            return NotImplemented
        cdef MatrixBase a_ = a
        cdef MatrixBase b_ = b
        if (a_.shape == (0, 0)):
            return b_
        if (b_.shape == (0, 0)):
            return a_
        if (a_.shape != b_.shape):
            raise ShapeError("Invalid shapes for matrix addition. Got %s %s" % (a_.shape, b_.shape))
        return a_.add_matrix(b_)

    def __mul__(a, b):
        a = _sympify(a, False)
        b = _sympify(b, False)
        if isinstance(a, MatrixBase):
            if isinstance(b, MatrixBase):
                if (a.ncols() != b.nrows()):
                    raise ShapeError("Invalid shapes for matrix multiplication. Got %s %s" % (a.shape, b.shape))
                return a.mul_matrix(b)
            elif isinstance(b, Basic):
                return a.mul_scalar(b)
            else:
                return NotImplemented
        elif isinstance(a, Basic):
            return b.mul_scalar(a)
        else:
            return NotImplemented

    def __truediv__(a, b):
        return div_matrices(a, b)

    def __sub__(a, b):
        a = _sympify(a, False)
        b = _sympify(b, False)
        if not isinstance(a, MatrixBase) or not isinstance(b, MatrixBase):
            return NotImplemented
        cdef MatrixBase a_ = a
        cdef MatrixBase b_ = b
        if (a_.shape != b_.shape):
            raise ShapeError("Invalid shapes for matrix subtraction. Got %s %s" % (a.shape, b.shape))
        return a_.add_matrix(-b_)

    def __neg__(self):
        return self.mul_scalar(-1)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if (self.ncols() == 0 or self.nrows() == 0):
                return []
            return [self.get(i // self.ncols(), i % self.ncols()) for i in range(*item.indices(len(self)))]
        elif isinstance(item, int):
            return self.get(item // self.ncols(), item % self.ncols())
        elif isinstance(item, tuple) and len(item) == 2:
            if is_sequence(item[0]) or is_sequence(item[1]):
                if isinstance(item[0], slice):
                    row_iter = range(*item[0].indices(self.rows))
                elif is_sequence(item[0]):
                    row_iter = item[0]
                else:
                    row_iter = [item[0]]

                if isinstance(item[1], slice):
                    col_iter = range(*item[1].indices(self.cols))
                elif is_sequence(item[1]):
                    col_iter = item[1]
                else:
                    col_iter = [item[1]]

                v = []
                for row in row_iter:
                    for col in col_iter:
                        v.append(self.get(row, col))
                return self.__class__(len(row_iter), len(col_iter), v)

            if isinstance(item[0], int) and isinstance(item[1], int):
                return self.get(item[0], item[1])
            else:
                s = [0, 0, 0, 0, 0, 0]
                for i in (0, 1):
                    if isinstance(item[i], slice):
                        s[i], s[i+2], s[i+4] = item[i].indices(self.nrows() if i == 0 else self.ncols())
                    else:
                        s[i], s[i+2], s[i+4] = item[i], item[i] + 1, 1
                if (s[0] < 0 or s[0] > self.rows or s[0] >= s[2] or s[2] < 0 or s[2] > self.rows):
                    raise IndexError
                if (s[1] < 0 or s[1] > self.cols or s[1] >= s[3] or s[3] < 0 or s[3] > self.cols):
                    raise IndexError
                return self._submatrix(*s)
        elif is_sequence(item):
            return [self.get(ind // self.ncols(), ind % self.ncols()) for ind in item]
        else:
            raise NotImplementedError

    def __setitem__(self, key, value):
        cdef unsigned k, l
        if isinstance(key, int):
            self.set(key // self.ncols(), key % self.ncols(), value)
        elif isinstance(key, slice):
            k = 0
            for i in range(*key.indices(len(self))):
                self.set(i // self.ncols(), i % self.ncols(), value[k])
                k = k + 1
        elif isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], slice):
                row_iter = range(*key[0].indices(self.rows))
            elif is_sequence(key[0]):
                row_iter = key[0]
            else:
                row_iter = [key[0]]

            if isinstance(key[1], slice):
                col_iter = range(*key[1].indices(self.cols))
            elif is_sequence(key[1]):
                col_iter = key[1]
            else:
                col_iter = [key[1]]

            for r, row in enumerate(row_iter):
                for c, col in enumerate(col_iter):
                    if not is_sequence(value):
                        self.set(row, col, value)
                        continue
                    try:
                        self.set(row, col, value[r, c])
                        continue
                    except TypeError:
                        pass
                    try:
                        self.set(row, col, value[r][c])
                        continue
                    except TypeError:
                        pass

                    if len(row_iter) == 1:
                        self.set(row, col, value[c])
                        continue

                    if len(col_iter) == 1:
                        self.set(row, col, value[r])
                        continue

        elif is_sequence(key) and is_sequence(value):
            for val, ind in zip(value, key):
                self.set(ind // self.ncols(), ind % self.ncols(), val)
        else:
            raise NotImplementedError

    def row_join(self, rhs):
        cdef DenseMatrixBase o = sympify(rhs)
        if self.rows != o.rows:
            raise ShapeError("`self` and `rhs` must have the same number of rows.")
        cdef DenseMatrixBase d = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(d.thisptr)).row_join(deref(symengine.static_cast_DenseMatrix(o.thisptr)))
        return d

    def col_join(self, bott):
        cdef DenseMatrixBase o = sympify(bott)
        if self.cols != o.cols:
            raise ShapeError("`self` and `rhs` must have the same number of columns.")
        cdef DenseMatrixBase d = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(d.thisptr)).col_join(deref(symengine.static_cast_DenseMatrix(o.thisptr)))
        return d

    def row_insert(self, pos, bott):
        cdef DenseMatrixBase o = sympify(bott)
        if pos < 0:
            pos = self.rows + pos
        if pos < 0:
            pos = 0
        elif pos > self.rows:
            pos = self.rows
        if self.cols != o.cols:
            raise ShapeError("`self` and `other` must have the same number of columns.")
        cdef DenseMatrixBase d = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(d.thisptr)).row_insert(deref(symengine.static_cast_DenseMatrix(o.thisptr)), pos)
        return d

    def col_insert(self, pos, bott):
        cdef DenseMatrixBase o = sympify(bott)
        if pos < 0:
            pos = self.cols + pos
        if pos < 0:
            pos = 0
        elif pos > self.cols:
            pos = self.cols
        if self.rows != o.rows:
            raise ShapeError("`self` and `other` must have the same number of rows.")
        cdef DenseMatrixBase d = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(d.thisptr)).col_insert(deref(symengine.static_cast_DenseMatrix(o.thisptr)), pos)
        return d

    def dot(self, b):
        cdef DenseMatrixBase o = sympify(b)
        cdef DenseMatrixBase result = self.__class__(self.rows, self.cols)
        symengine.dot(deref(symengine.static_cast_DenseMatrix(self.thisptr)), deref(symengine.static_cast_DenseMatrix(o.thisptr)), deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        if len(result) == 1:
            return result[0, 0]
        else:
            return result

    def cross(self, b):
        cdef DenseMatrixBase o = sympify(b)
        if self.cols * self.rows != 3 or o.cols * o.rows != 3:
            raise ShapeError("Dimensions incorrect for cross product: %s x %s" %
                             ((self.rows, self.cols), (b.rows, b.cols)))
        cdef DenseMatrixBase result = self.__class__(self.rows, self.cols)
        symengine.cross(deref(symengine.static_cast_DenseMatrix(self.thisptr)), deref(symengine.static_cast_DenseMatrix(o.thisptr)), deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        return result

    @property
    def rows(self):
        return self.nrows()

    @property
    def cols(self):
        return self.ncols()

    @property
    def is_square(self):
        return deref(self.thisptr).is_square()

    def nrows(self):
        return deref(self.thisptr).nrows()

    def ncols(self):
        return deref(self.thisptr).ncols()

    def __len__(self):
        return self.nrows() * self.ncols()

    property shape:
        def __get__(self):
            return (self.nrows(), self.ncols())

    property size:
        def __get__(self):
            return self.nrows()*self.ncols()

    def ravel(self, order='C'):
        if order == 'C':
            return [self._get(i, j) for i in range(self.nrows())
                    for j in range(self.ncols())]
        elif order == 'F':
            return [self._get(i, j) for j in range(self.ncols())
                    for i in range(self.nrows())]
        else:
            raise NotImplementedError("Unknown order '%s'" % order)

    def reshape(self, rows, cols):
        if len(self) != rows*cols:
            raise ValueError("Invalid reshape parameters %d %d" % (rows, cols))
        cdef DenseMatrixBase r = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(r.thisptr)).resize(rows, cols)
        return r

    def _get_index(self, i, j):
        nr = self.nrows()
        nc = self.ncols()
        if i < 0:
            i += nr
        if j < 0:
            j += nc
        if i < 0 or i >= nr:
            raise IndexError("Row index out of bounds: %d" % i)
        if j < 0 or j >= nc:
            raise IndexError("Column index out of bounds: %d" % j)
        return i, j

    def get(self, i, j):
        i, j = self._get_index(i, j)
        return self._get(i, j)

    def _get(self, i, j):
        # No error checking is done
        return c2py(deref(self.thisptr).get(i, j))

    def col(self, j):
        return self[:, j]

    def row(self, i):
        return self[i, :]

    def set(self, i, j, e):
        i, j = self._get_index(i, j)
        return self._set(i, j, e)

    def _set(self, i, j, e):
        # No error checking is done
        cdef Basic e_ = sympify(e)
        if e_ is not None:
            deref(self.thisptr).set(i, j, e_.thisptr)

    def det(self):
        if self.nrows() != self.ncols():
            raise NonSquareMatrixError()
        return c2py(deref(self.thisptr).det())

    def inv(self, method='LU'):
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())

        if method.upper() == 'LU':
            ## inv() method of DenseMatrixBase uses LU factorization
            deref(self.thisptr).inv(deref(result.thisptr))
        elif method.upper() == 'FFLU':
            symengine.inverse_FFLU(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        elif method.upper() == 'GJ':
            symengine.inverse_GJ(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        else:
            raise Exception("Unsupported method.")
        return result

    def add_matrix(self, A):
        cdef MatrixBase A_ = sympify(A)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).add_matrix(deref(A_.thisptr), deref(result.thisptr))
        return result

    def mul_matrix(self, A):
        cdef MatrixBase A_ = sympify(A)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), A.ncols())
        deref(self.thisptr).mul_matrix(deref(A_.thisptr), deref(result.thisptr))
        return result

    def multiply_elementwise(self, A):
        cdef MatrixBase A_ = sympify(A)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).elementwise_mul_matrix(deref(A_.thisptr), deref(result.thisptr))
        return result

    def add_scalar(self, k):
        cdef Basic k_ = sympify(k)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).add_scalar(k_.thisptr, deref(result.thisptr))
        return result

    def mul_scalar(self, k):
        cdef Basic k_ = sympify(k)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).mul_scalar(k_.thisptr, deref(result.thisptr))
        return result

    def transpose(self):
        cdef DenseMatrixBase result = self.__class__(self.ncols(), self.nrows())
        deref(self.thisptr).transpose(deref(result.thisptr))
        return result

    def conjugate(self):
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).conjugate(deref(result.thisptr))
        return result

    def conjugate_transpose(self):
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).conjugate_transpose(deref(result.thisptr))
        return result

    @property
    def H(self):
        return self.conjugate_transpose()

    def trace(self):
        return c2py(deref(symengine.static_cast_DenseMatrix(self.thisptr)).trace())

    @property
    def is_zero_matrix(self):
        return tribool(deref(symengine.static_cast_DenseMatrix(self.thisptr)).is_zero())

    @property
    def is_real_matrix(self):
        return tribool(deref(symengine.static_cast_DenseMatrix(self.thisptr)).is_real())

    @property
    def is_diagonal(self):
        return tribool(deref(symengine.static_cast_DenseMatrix(self.thisptr)).is_diagonal())

    @property
    def is_symmetric(self):
        return tribool(deref(symengine.static_cast_DenseMatrix(self.thisptr)).is_symmetric())

    @property
    def is_hermitian(self):
        return tribool(deref(symengine.static_cast_DenseMatrix(self.thisptr)).is_hermitian())

    @property
    def is_weakly_diagonally_dominant(self):
        return tribool(deref(symengine.static_cast_DenseMatrix(self.thisptr)).is_weakly_diagonally_dominant())

    @property
    def is_strongly_diagonally_dominant(self):
        return tribool(deref(symengine.static_cast_DenseMatrix(self.thisptr)).is_strictly_diagonally_dominant())

    @property
    def T(self):
        return self.transpose()

    def applyfunc(self, f):
        cdef DenseMatrixBase out = self.__class__(self)
        cdef int nr = self.nrows()
        cdef int nc = self.ncols()
        cdef Basic e_;
        for i in range(nr):
            for j in range(nc):
                e_ = sympify(f(self._get(i, j)))
                if e_ is not None:
                    deref(out.thisptr).set(i, j, e_.thisptr)
        return out

    def _applyfunc(self, f):
        cdef int nr = self.nrows()
        cdef int nc = self.ncols()
        for i in range(nr):
            for j in range(nc):
                self._set(i, j, f(self._get(i, j)))

    def msubs(self, *args):
        cdef _DictBasic D = get_dict(*args)
        return self.applyfunc(lambda x: x.msubs(D))

    def _diff(self, Basic x):
        cdef DenseMatrixBase R = self.__class__(self.rows, self.cols)
        symengine.diff(<const symengine.DenseMatrix &>deref(self.thisptr),
                x.thisptr, <symengine.DenseMatrix &>deref(R.thisptr))
        return R

    def diff(self, *args):
        return diff(self, *args)

    #TODO: implement this in C++
    def subs(self, *args):
        cdef _DictBasic D = get_dict(*args)
        return self.applyfunc(lambda x: x.subs(D))

    def xreplace(self, *args):
        cdef _DictBasic D = get_dict(*args)
        return self.applyfunc(lambda x: x.xreplace(D))

    replace = xreplace

    @property
    def free_symbols(self):
        cdef symengine.set_basic _set = symengine.free_symbols(deref(self.thisptr))
        return {c2py(<rcp_const_basic>(elem)) for elem in _set}

    def _submatrix(self, unsigned r_i, unsigned c_i, unsigned r_j, unsigned c_j, unsigned r_s=1, unsigned c_s=1):
        r_j, c_j = r_j - 1, c_j - 1
        cdef DenseMatrixBase result = self.__class__(((r_j - r_i) // r_s) + 1, ((c_j - c_i) // c_s) + 1)
        deref(self.thisptr).submatrix(deref(result.thisptr), r_i, c_i, r_j, c_j, r_s, c_s)
        return result

    def LU(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase U = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).LU(deref(L.thisptr), deref(U.thisptr))
        return L, U

    def LDL(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase D = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).LDL(deref(L.thisptr), deref(D.thisptr))
        return L, D

    def solve(self, b, method='LU'):
        cdef DenseMatrixBase b_ = sympify(b)
        cdef DenseMatrixBase x = self.__class__(b_.nrows(), b_.ncols())

        if method.upper() == 'LU':
            ## solve() method of DenseMatrixBase uses LU factorization
            symengine.pivoted_LU_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        elif method.upper() == 'FFLU':
            symengine.FFLU_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        elif method.upper() == 'LDL':
            symengine.LDL_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        elif method.upper() == 'FFGJ':
            symengine.FFGJ_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        else:
            raise Exception("Unsupported method.")

        return x

    def LUsolve(self, b):
        cdef DenseMatrixBase b_ = sympify(b)
        cdef DenseMatrixBase x = self.__class__(b.nrows(), b.ncols())
        symengine.pivoted_LU_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
            deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
            deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        return x

    def FFLU(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase U = self.__class__(self.nrows(), self.ncols(), [0]*self.nrows()*self.ncols())
        deref(self.thisptr).FFLU(deref(L.thisptr))

        for i in range(self.nrows()):
            for j in range(i + 1, self.ncols()):
                U.set(i, j, L.get(i, j))
                L.set(i, j, 0)
            U.set(i, i, L.get(i, i))

        return L, U

    def FFLDU(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase D = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase U = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).FFLDU(deref(L.thisptr), deref(D.thisptr), deref(U.thisptr))
        return L, D, U

    def QR(self):
        cdef DenseMatrixBase Q = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase R = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).QR(deref(Q.thisptr), deref(R.thisptr))
        return Q, R

    def cholesky(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).cholesky(deref(L.thisptr))
        return L

    def jacobian(self, x):
        cdef DenseMatrixBase x_ = sympify(x)
        cdef DenseMatrixBase R = self.__class__(self.nrows(), x.nrows())
        symengine.jacobian(<const symengine.DenseMatrix &>deref(self.thisptr),
                <const symengine.DenseMatrix &>deref(x_.thisptr),
                <symengine.DenseMatrix &>deref(R.thisptr))
        return R

    def _sympy_(self):
        s = []
        cdef symengine.DenseMatrix A = deref(symengine.static_cast_DenseMatrix(self.thisptr))
        for i in range(A.nrows()):
            l = []
            for j in range(A.ncols()):
                l.append(c2py(A.get(i, j))._sympy_())
            s.append(l)
        import sympy
        return sympy.Matrix(s)

    def _sage_(self):
        s = []
        cdef symengine.DenseMatrix A = deref(symengine.static_cast_DenseMatrix(self.thisptr))
        for i in range(A.nrows()):
            l = []
            for j in range(A.ncols()):
                l.append(c2py(A.get(i, j))._sage_())
            s.append(l)
        import sage.all as sage
        return sage.Matrix(s)

    def dump_real(self, double[::1] out):
        cdef size_t ri, ci, nr, nc
        if out.size < self.size:
            raise ValueError("out parameter too short")
        nr = self.nrows()
        nc = self.ncols()
        for ri in range(nr):
            for ci in range(nc):
                out[ri*nc + ci] = symengine.eval_double(deref(
                    <symengine.rcp_const_basic>(deref(self.thisptr).get(ri, ci))))

    def dump_complex(self, double complex[::1] out):
        cdef size_t ri, ci, nr, nc
        if out.size < self.size:
            raise ValueError("out parameter too short")
        nr = self.nrows()
        nc = self.ncols()
        for ri in range(nr):
            for ci in range(nc):
                out[ri*nc + ci] = symengine.eval_complex_double(deref(
                    <symengine.rcp_const_basic>(deref(self.thisptr).get(ri, ci))))

    def __iter__(self):
        return DenseMatrixBaseIter(self)

    def as_mutable(self):
        return MutableDenseMatrix(self)

    def as_immutable(self):
        return ImmutableDenseMatrix(self)

    def tolist(self):
        return [[self[rowi, coli] for coli in range(self.ncols())]
                for rowi in range(self.nrows())]

    def __array__(self):
        return np.array(self.tolist())

    def _mat(self):
        return self

    def atoms(self, *types):
        if types:
            s = set()
            if (isinstance(self, types)):
                s.add(self)
            for arg in self:
                s.update(arg.atoms(*types))
            return s
        else:
           return self.free_symbols

    def simplify(self, *args, **kwargs):
        return self._applyfunc(lambda x : x.simplify(*args, **kwargs))

    def expand(self, *args, **kwargs):
        return self.applyfunc(lambda x : x.expand(*args, **kwargs))


def div_matrices(a, b):
    a = _sympify(a, False)
    b = _sympify(b, False)
    if isinstance(a, MatrixBase):
        if isinstance(b, MatrixBase):
            return a.mul_matrix(b.inv())
        elif isinstance(b, Basic):
            return a.mul_scalar(1/b)
        else:
            return NotImplemented
    else:
        return NotImplemented

class DenseMatrixBaseIter(object):

    def __init__(self, d):
        self.curr = -1
        self.d = d

    def __iter__(self):
        return self

    def __next__(self):
        self.curr = self.curr + 1
        if (self.curr < self.d.rows * self.d.cols):
            return self.d._get(self.curr // self.d.cols, self.curr % self.d.cols)
        else:
            raise StopIteration

    next = __next__

cdef class MutableDenseMatrix(DenseMatrixBase):

    def col_swap(self, i, j):
        symengine.column_exchange_dense(deref(symengine.static_cast_DenseMatrix(self.thisptr)), i, j)

    def fill(self, value):
        for i in range(self.rows):
            for j in range(self.cols):
                self[i, j] = value

    def row_swap(self, i, j):
        symengine.row_exchange_dense(deref(symengine.static_cast_DenseMatrix(self.thisptr)), i, j)

    def rowmul(self, i, c, *args):
        cdef Basic _c = sympify(c)
        symengine.row_mul_scalar_dense(deref(symengine.static_cast_DenseMatrix(self.thisptr)), i, _c.thisptr)
        return self

    def rowadd(self, i, j, c, *args):
        cdef Basic _c = sympify(c)
        symengine.row_add_row_dense(deref(symengine.static_cast_DenseMatrix(self.thisptr)), i, j, _c.thisptr)
        return self

    def row_del(self, i):
        if i < -self.rows or i >= self.rows:
            raise IndexError("Index out of range: 'i = %s', valid -%s <= i"
                             " < %s" % (i, self.rows, self.rows))
        if i < 0:
            i += self.rows
        deref(symengine.static_cast_DenseMatrix(self.thisptr)).row_del(i)
        return self

    def col_del(self, i):
        if i < -self.cols or i >= self.cols:
            raise IndexError("Index out of range: 'i=%s', valid -%s <= i < %s"
                             % (i, self.cols, self.cols))
        if i < 0:
            i += self.cols
        deref(symengine.static_cast_DenseMatrix(self.thisptr)).col_del(i)
        return self

Matrix = DenseMatrix = MutableDenseMatrix

cdef class ImmutableDenseMatrix(DenseMatrixBase):

    def __setitem__(self, key, value):
        raise TypeError("Cannot set values of {}".format(self.__class__))

ImmutableMatrix = ImmutableDenseMatrix


cdef matrix_to_vec(DenseMatrixBase d, symengine.vec_basic& v):
    cdef Basic e_
    for i in range(d.nrows()):
        for j in range(d.ncols()):
            e_ = d._get(i, j)
            v.push_back(e_.thisptr)


def eye(n):
    cdef DenseMatrixBase d = DenseMatrix(n, n)
    symengine.eye(deref(symengine.static_cast_DenseMatrix(d.thisptr)), 0)
    return d


cdef symengine.vec_basic iter_to_vec_basic(iter):
    cdef Basic B
    cdef symengine.vec_basic V
    for b in iter:
        B = sympify(b)
        V.push_back(B.thisptr)
    return V


def diag(*values):
    cdef DenseMatrixBase d = DenseMatrix(len(values), len(values))
    cdef symengine.vec_basic V = iter_to_vec_basic(values)
    symengine.diag(deref(symengine.static_cast_DenseMatrix(d.thisptr)), V, 0)
    return d


def ones(r, c = None):
    if c is None:
        c = r
    cdef DenseMatrixBase d = DenseMatrix(r, c)
    symengine.ones(deref(symengine.static_cast_DenseMatrix(d.thisptr)))
    return d


def zeros(r, c = None):
    if c is None:
        c = r
    cdef DenseMatrixBase d = DenseMatrix(r, c)
    symengine.zeros(deref(symengine.static_cast_DenseMatrix(d.thisptr)))
    return d


cdef class Sieve:
    @staticmethod
    def generate_primes(n):
        cdef symengine.vector[unsigned] primes
        symengine.sieve_generate_primes(primes, n)
        s = []
        for i in range(primes.size()):
            s.append(primes[i])
        return s


cdef class Sieve_iterator:
    cdef symengine.sieve_iterator *thisptr
    cdef unsigned limit
    def __cinit__(self):
        self.thisptr = new symengine.sieve_iterator()
        self.limit = 0

    def __cinit__(self, n):
        self.thisptr = new symengine.sieve_iterator(n)
        self.limit = n

    def __iter__(self):
        return self

    def __next__(self):
        n = deref(self.thisptr).next_prime()
        if self.limit > 0 and n > self.limit:
            raise StopIteration
        else:
            return n


def module_cleanup():
    global I, E, pi, oo, minus_oo, zoo, nan, true, false, golden_ratio, \
           catalan, eulergamma, sympy_module, sage_module, half, one, \
           minus_one, zero
    funcs.clear()
    del    I, E, pi, oo, minus_oo, zoo, nan, true, false, golden_ratio, \
           catalan, eulergamma, sympy_module, sage_module, half, one, \
           minus_one, zero

import atexit
atexit.register(module_cleanup)

def diff(ex, *args):
    ex = sympify(ex)
    prev = 0
    cdef Basic b
    cdef size_t i
    for x in args:
        b = sympify(x)
        if isinstance(b, Integer):
            i = int(b) - 1
            for j in range(i):
                ex = ex._diff(prev)
        else:
            ex = ex._diff(b)
        prev = b
    return ex

def expand(x, deep=True):
    return sympify(x).expand(deep)

expand_mul = expand

def function_symbol(name, *args):
    cdef symengine.vec_basic v
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        if e_ is not None:
            v.push_back(e_.thisptr)
    return c2py(symengine.function_symbol(name.encode("utf-8"), v))

def sqrt(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.sqrt(X.thisptr))

def exp(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.exp(X.thisptr))

def perfect_power(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return symengine.perfect_power(deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def is_square(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return symengine.perfect_square(deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def integer_nthroot(a, n):
    cdef Basic _a = sympify(a)
    require(_a, Integer)
    cdef RCP[const symengine.Integer] _r
    cdef int ret_val = symengine.i_nth_root(symengine.outArg_Integer(_r), deref(symengine.rcp_static_cast_Integer(_a.thisptr)), n)
    return (c2py(<rcp_const_basic>_r), ret_val == 1)

def _max(*args):
    cdef symengine.vec_basic v
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        v.push_back(e_.thisptr)
    return c2py(symengine.max(v))

def _min(*args):
    cdef symengine.vec_basic v
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        v.push_back(e_.thisptr)
    return c2py(symengine.min(v))

def gamma(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.gamma(X.thisptr))

def eq(lhs, rhs = None):
    cdef Basic X = sympify(lhs)
    if rhs is None:
        return c2py(<rcp_const_basic>(symengine.Eq(X.thisptr)))
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Eq(X.thisptr, Y.thisptr)))

def ne(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Ne(X.thisptr, Y.thisptr)))

def ge(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Ge(X.thisptr, Y.thisptr)))

Ge = GreaterThan = ge

def gt(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Gt(X.thisptr, Y.thisptr)))

Gt = StrictGreaterThan = gt

def le(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Le(X.thisptr, Y.thisptr)))

def lt(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Lt(X.thisptr, Y.thisptr)))

def digamma(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.digamma(X.thisptr))

def trigamma(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.trigamma(X.thisptr))

def logical_and(*args):
    cdef symengine.set_boolean s
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_and(s)))

def logical_or(*args):
    cdef symengine.set_boolean s
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_or(s)))

def Nor(*args):
    cdef symengine.set_boolean s
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_nor(s)))

def Nand(*args):
    cdef symengine.set_boolean s
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_nand(s)))

def logical_not(x):
    cdef Basic x_ = sympify(x)
    require(x_, Boolean)
    cdef RCP[const symengine.Boolean] _x = symengine.rcp_static_cast_Boolean(x_.thisptr)
    return c2py(<rcp_const_basic>(symengine.logical_not(_x)))

def logical_xor(*args):
    cdef symengine.vec_boolean v
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        v.push_back(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_xor(v)))

def Xnor(*args):
    cdef symengine.vec_boolean v
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        v.push_back(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_xnor(v)))

def evalf(x, unsigned long bits=53, real=None):
    cdef Basic X = sympify(x)
    cdef symengine.EvalfDomain d
    if real is None:
        d = symengine.EvalfSymbolic
    elif real:
        d = symengine.EvalfReal
    else:
        d = symengine.EvalfComplex
    return c2py(<rcp_const_basic>(symengine.evalf(deref(X.thisptr), bits, d)))

def eval_double(x):
    warnings.warn("eval_double is deprecated. Use evalf(..., real=True)", DeprecationWarning)
    return evalf(x, 53, real=True)

def eval_complex_double(x):
    warnings.warn("eval_complex_double is deprecated. Use evalf(..., real=False)", DeprecationWarning)
    return evalf(x, 53, real=False)

have_mpfr = False
have_mpc = False
have_piranha = False
have_flint = False
have_llvm = False
have_llvm_long_double = False

IF HAVE_SYMENGINE_MPFR:
    have_mpfr = True
    def eval_mpfr(x, unsigned long prec):
        warnings.warn("eval_mpfr is deprecated. Use evalf(..., real=True)", DeprecationWarning)
        return evalf(x, prec, real=True)

IF HAVE_SYMENGINE_MPC:
    have_mpc = True
    def eval_mpc(x, unsigned long prec):
        warnings.warn("eval_mpc is deprecated. Use evalf(..., real=False)", DeprecationWarning)
        return evalf(x, prec, real=False)

IF HAVE_SYMENGINE_PIRANHA:
    have_piranha = True

IF HAVE_SYMENGINE_FLINT:
    have_flint = True

IF HAVE_SYMENGINE_LLVM:
    have_llvm = True

IF HAVE_SYMENGINE_LLVM_LONG_DOUBLE:
    have_llvm_long_double = True

def require(obj, t):
    if not isinstance(obj, t):
        raise TypeError("{} required. {} is of type {}".format(t, obj, type(obj)))

def eval(x, long prec):
    warnings.warn("eval is deprecated. Use evalf(..., real=False)", DeprecationWarning)
    return evalf(x, prec, real=False)

def eval_real(x, long prec):
    warnings.warn("eval_real is deprecated. Use evalf(..., real=True)", DeprecationWarning)
    return evalf(x, prec, real=True)

def probab_prime_p(n, reps = 25):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return symengine.probab_prime_p(deref(symengine.rcp_static_cast_Integer(_n.thisptr)), reps) >= 1

isprime = probab_prime_p

def nextprime(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return c2py(<rcp_const_basic>(symengine.nextprime(deref(symengine.rcp_static_cast_Integer(_n.thisptr)))))

def gcd(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<rcp_const_basic>(symengine.gcd(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def lcm(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<rcp_const_basic>(symengine.lcm(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def gcd_ext(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    cdef RCP[const symengine.Integer] g, s, t
    symengine.gcd_ext(symengine.outArg_Integer(g), symengine.outArg_Integer(s), symengine.outArg_Integer(t),
        deref(symengine.rcp_static_cast_Integer(_a.thisptr)), deref(symengine.rcp_static_cast_Integer(_b.thisptr)))
    return (c2py(<rcp_const_basic>s), c2py(<rcp_const_basic>t), c2py(<rcp_const_basic>g))

igcdex = gcd_ext

def mod(a, b):
    if b == 0:
        raise ZeroDivisionError
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<rcp_const_basic>(symengine.mod(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def quotient(a, b):
    if b == 0:
        raise ZeroDivisionError
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<rcp_const_basic>(symengine.quotient(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def quotient_mod(a, b):
    if b == 0:
        raise ZeroDivisionError
    cdef RCP[const symengine.Integer] q, r
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    symengine.quotient_mod(symengine.outArg_Integer(q), symengine.outArg_Integer(r),
        deref(symengine.rcp_static_cast_Integer(_a.thisptr)), deref(symengine.rcp_static_cast_Integer(_b.thisptr)))
    return (c2py(<rcp_const_basic>q), c2py(<rcp_const_basic>r))

def mod_inverse(a, b):
    cdef RCP[const symengine.Integer] inv
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    cdef int ret_val = symengine.mod_inverse(symengine.outArg_Integer(inv),
        deref(symengine.rcp_static_cast_Integer(_a.thisptr)), deref(symengine.rcp_static_cast_Integer(_b.thisptr)))
    if ret_val == 0:
        return None
    return c2py(<rcp_const_basic>inv)

def crt(rem, mod):
    cdef symengine.vec_integer _rem, _mod
    cdef Basic _a
    cdef cppbool ret_val
    for i in range(len(rem)):
        _a = sympify(rem[i])
        require(_a, Integer)
        _rem.push_back(symengine.rcp_static_cast_Integer(_a.thisptr))
        _a = sympify(mod[i])
        require(_a, Integer)
        _mod.push_back(symengine.rcp_static_cast_Integer(_a.thisptr))

    cdef RCP[const symengine.Integer] c
    ret_val = symengine.crt(symengine.outArg_Integer(c), _rem, _mod)
    if not ret_val:
        return None
    return c2py(<rcp_const_basic>c)

def fibonacci(n):
    if n < 0 :
        raise NotImplementedError
    return c2py(<rcp_const_basic>(symengine.fibonacci(n)))

def fibonacci2(n):
    if n < 0 :
        raise NotImplementedError
    cdef RCP[const symengine.Integer] f1, f2
    symengine.fibonacci2(symengine.outArg_Integer(f1), symengine.outArg_Integer(f2), n)
    return [c2py(<rcp_const_basic>f1), c2py(<rcp_const_basic>f2)]

def lucas(n):
    if n < 0 :
        raise NotImplementedError
    return c2py(<rcp_const_basic>(symengine.lucas(n)))

def lucas2(n):
    if n < 0 :
        raise NotImplementedError
    cdef RCP[const symengine.Integer] f1, f2
    symengine.lucas2(symengine.outArg_Integer(f1), symengine.outArg_Integer(f2), n)
    return [c2py(<rcp_const_basic>f1), c2py(<rcp_const_basic>f2)]

def binomial(n, k):
    if k < 0:
        raise ArithmeticError
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return c2py(<rcp_const_basic>symengine.binomial(deref(symengine.rcp_static_cast_Integer(_n.thisptr)), k))

def factorial(n):
    if n < 0:
        raise ArithmeticError
    return c2py(<rcp_const_basic>(symengine.factorial(n)))

def divides(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return symengine.divides(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))

def factor(n, B1 = 1.0):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)), B1)
    if (ret_val == 1):
        return c2py(<rcp_const_basic>f)
    else:
        return None

def factor_lehman_method(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor_lehman_method(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    if (ret_val == 1):
        return c2py(<rcp_const_basic>f)
    else:
        return None

def factor_pollard_pm1_method(n, B = 10, retries = 5):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor_pollard_pm1_method(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)), B, retries)
    if (ret_val == 1):
        return c2py(<rcp_const_basic>f)
    else:
        return None

def factor_pollard_rho_method(n, retries = 5):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor_pollard_rho_method(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)), retries)
    if (ret_val == 1):
        return c2py(<rcp_const_basic>f)
    else:
        return None

def prime_factors(n):
    cdef symengine.vec_integer factors
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    symengine.prime_factors(factors, deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    s = []
    for i in range(factors.size()):
        s.append(c2py(<rcp_const_basic>(factors[i])))
    return s

def prime_factor_multiplicities(n):
    cdef symengine.vec_integer factors
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    symengine.prime_factors(factors, deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    cdef Basic r
    dict = {}
    for i in range(factors.size()):
        r = c2py(<rcp_const_basic>(factors[i]))
        if (r not in dict):
            dict[r] = 1
        else:
            dict[r] += 1
    return dict

def bernoulli(n):
    if n < 0:
        raise ArithmeticError
    return c2py(<rcp_const_basic>(symengine.bernoulli(n)))

def primitive_root(n):
    cdef RCP[const symengine.Integer] g
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef cppbool ret_val = symengine.primitive_root(symengine.outArg_Integer(g),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    if ret_val == 0:
        return None
    return c2py(<rcp_const_basic>g)

def primitive_root_list(n):
    cdef symengine.vec_integer root_list
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    symengine.primitive_root_list(root_list,
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    s = []
    for i in range(root_list.size()):
        s.append(c2py(<rcp_const_basic>(root_list[i])))
    return s

def totient(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] m = symengine.rcp_static_cast_Integer(_n.thisptr)
    return c2py(<rcp_const_basic>symengine.totient(m))

def carmichael(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] m = symengine.rcp_static_cast_Integer(_n.thisptr)
    return c2py(<rcp_const_basic>symengine.carmichael(m))

def multiplicative_order(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    cdef RCP[const symengine.Integer] n1 = symengine.rcp_static_cast_Integer(_n.thisptr)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] o
    cdef cppbool c = symengine.multiplicative_order(symengine.outArg_Integer(o),
        a1, n1)
    if not c:
        return None
    return c2py(<rcp_const_basic>o)

def legendre(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    return symengine.legendre(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def jacobi(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    return symengine.jacobi(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def kronecker(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    return symengine.kronecker(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def nthroot_mod(a, n, m):
    cdef RCP[const symengine.Integer] root
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    require(_n, Integer)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] n1 = symengine.rcp_static_cast_Integer(_n.thisptr)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    cdef cppbool ret_val = symengine.nthroot_mod(symengine.outArg_Integer(root), a1, n1, m1)
    if not ret_val:
        return None
    return c2py(<rcp_const_basic>root)

def nthroot_mod_list(a, n, m):
    cdef symengine.vec_integer root_list
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    require(_n, Integer)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] n1 = symengine.rcp_static_cast_Integer(_n.thisptr)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    symengine.nthroot_mod_list(root_list, a1, n1, m1)
    s = []
    for i in range(root_list.size()):
        s.append(c2py(<rcp_const_basic>(root_list[i])))
    return s

def sqrt_mod(a, p, all_roots=False):
    if all_roots:
        return nthroot_mod_list(a, 2, p)
    return nthroot_mod(a, 2, p)

def powermod(a, b, m):
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    cdef Number _b = sympify(b)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    cdef RCP[const symengine.Number] b1 = symengine.rcp_static_cast_Number(_b.thisptr)
    cdef RCP[const symengine.Integer] root
    cdef cppbool ret_val = symengine.powermod(symengine.outArg_Integer(root), a1, b1, m1)
    if ret_val == 0:
        return None
    return c2py(<rcp_const_basic>root)

def powermod_list(a, b, m):
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    cdef Number _b = sympify(b)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    cdef RCP[const symengine.Number] b1 = symengine.rcp_static_cast_Number(_b.thisptr)
    cdef symengine.vec_integer v

    symengine.powermod_list(v, a1, b1, m1)
    s = []
    for i in range(v.size()):
        s.append(c2py(<rcp_const_basic>(v[i])))
    return s

def has_symbol(obj, symbol=None):
    cdef Basic b = _sympify(obj)
    cdef Basic s = _sympify(symbol)
    require(s, Symbol)
    if (not symbol):
        return not b.free_symbols.empty()
    else:
        return symengine.has_symbol(deref(b.thisptr),
                deref(symengine.rcp_static_cast_Symbol(s.thisptr)))


cdef class _Lambdify(object):
    def __init__(self, args, *exprs, cppbool real=True, order='C', cppbool cse=False, cppbool _load=False, dtype=None, **kwargs):
        cdef:
            Basic e_
            size_t ri, ci, nr, nc
            symengine.MatrixBase *mtx
            rcp_const_basic b_
            symengine.vec_basic args_, outs_
            vector[int] out_sizes

        if _load:
            self.args_size, self.tot_out_size, self.out_shapes, self.real, \
                self.n_exprs, self.order, self.accum_out_sizes, self.numpy_dtype, \
                llvm_function = args
            self._load(llvm_function)
            return

        args = np.asanyarray(args)
        self.args_size = args.size
        exprs = tuple(np.asanyarray(expr) for expr in exprs)
        self.out_shapes = [expr.shape for expr in exprs]
        self.n_exprs = len(exprs)
        self.real = real
        self.order = order
        self.numpy_dtype = dtype if dtype else (np.float64 if self.real else np.complex128)
        if self.args_size == 0:
            raise NotImplementedError("Support for zero arguments not yet supported")
        self.tot_out_size = 0
        for idx, shape in enumerate(self.out_shapes):
            out_sizes.push_back(reduce(mul, shape or (1,)))
            self.tot_out_size += out_sizes[idx]
        for i in range(self.n_exprs + 1):
            self.accum_out_sizes.push_back(0)
            for j in range(i):
                self.accum_out_sizes[i] += out_sizes[j]

        for arg in np.ravel(args, order=self.order):
            e_ = _sympify(arg)
            args_.push_back(e_.thisptr)

        for curr_expr in exprs:
            if curr_expr.ndim == 0:
                e_ = _sympify(curr_expr.item())
                outs_.push_back(e_.thisptr)
            else:
                for e in np.ravel(curr_expr, order=self.order):
                    e_ = _sympify(e)
                    outs_.push_back(e_.thisptr)
        self._init(args_, outs_, cse)

    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse):
        raise ValueError("Not supported")

    cdef _load(self, const string &s):
        raise ValueError("Not supported")

    cpdef eval_real(self, inp, out):
        if inp.size != self.args_size:
            raise ValueError("Size of inp incompatible with number of args.")
        if out.size != self.tot_out_size:
            raise ValueError("Size of out incompatible with number of exprs.")
        self.unsafe_real(inp, out)

    cpdef eval_complex(self, inp, out):
        if inp.size != self.args_size:
            raise ValueError("Size of inp incompatible with number of args.")
        if out.size != self.tot_out_size:
            raise ValueError("Size of out incompatible with number of exprs.")
        self.unsafe_complex(inp, out)

    cpdef unsafe_eval(self, inp, out, unsigned nbroadcast=1):
        raise ValueError("Not supported")

    def __call__(self, *args, out=None):
        """
        Parameters
        ----------
        inp: array_like
            last dimension must be equal to number of arguments.
        out: array_like or None (default)
            Allows for low-overhead use (output argument, must be contiguous).
            If ``None``: an output container will be allocated (NumPy ndarray).
            If ``len(exprs) > 0`` output is found in the corresponding
            order.

        Returns
        -------
        If ``len(exprs) == 1``: ``numpy.ndarray``, otherwise a tuple of such.

        """
        cdef:
            size_t idx, new_tot_out_size, nbroadcast = 1
        if self.order not in ('C', 'F'):
            raise NotImplementedError("Only C & F order supported for now.")

        if len(args) == 1:
            args = args[0]

        try:
            inp = np.asanyarray(args, dtype=self.numpy_dtype)
        except TypeError:
            inp = np.fromiter(args, dtype=self.numpy_dtype)

        if inp.size < self.args_size or inp.size % self.args_size != 0:
            raise ValueError("Broadcasting failed (input/arg size mismatch)")
        nbroadcast = inp.size // self.args_size

        if inp.ndim > 1:
            if self.args_size > 1:
                if self.order == 'C':
                    if inp.shape[inp.ndim-1] != self.args_size:
                        raise ValueError(("C order implies last dim (%d) == len(args)"
                                          " (%d)") % (inp.shape[inp.ndim-1], self.args_size))
                    extra_dim = inp.shape[:inp.ndim-1]
                elif self.order == 'F':
                    if inp.shape[0] != self.args_size:
                        raise ValueError("F order implies first dim (%d) == len(args) (%d)"
                                         % (inp.shape[0], self.args_size))
                    extra_dim = inp.shape[1:]
            else:
                extra_dim = inp.shape
        else:
            if nbroadcast > 1 and inp.ndim == 1:
                extra_dim = (nbroadcast,)  # special case
            else:
                extra_dim = ()
        extra_left = extra_dim if self.order == 'C' else ()
        extra_right = () if self.order == 'C' else extra_dim
        new_out_shapes = [extra_left + out_shape + extra_right
                          for out_shape in self.out_shapes]

        new_tot_out_size = nbroadcast * self.tot_out_size
        if out is None:
            out = np.empty(new_tot_out_size, dtype=self.numpy_dtype, order=self.order)
        else:
            if out.size < new_tot_out_size:
                raise ValueError("Incompatible size of output argument")
            if out.ndim > 1:
                if len(self.out_shapes) > 1:
                    raise ValueError("output array with ndim > 1 assumes one output")
                out_shape, = self.out_shapes
                if self.order == 'C':
                    if not out.flags['C_CONTIGUOUS']:
                        raise ValueError("Output argument needs to be C-contiguous")
                    if out.shape[-len(out_shape):] != tuple(out_shape):
                        raise ValueError("shape mismatch for output array")
                elif self.order == 'F':
                    if not out.flags['F_CONTIGUOUS']:
                        raise ValueError("Output argument needs to be F-contiguous")
                    if out.shape[:len(out_shape)] != tuple(out_shape):
                        raise ValueError("shape mismatch for output array")
            else:
                if not out.flags['F_CONTIGUOUS']:  # or C_CONTIGUOUS (ndim <= 1)
                    raise ValueError("Output array need to be contiguous")
            if not out.flags['WRITEABLE']:
                raise ValueError("Output argument needs to be writeable")
            out = out.ravel(order=self.order)

        self.unsafe_eval(inp, out, nbroadcast)

        if self.order == 'C':
            out = out.reshape((nbroadcast, self.tot_out_size), order='C')
            result = [
                out[:, self.accum_out_sizes[idx]:self.accum_out_sizes[idx+1]].reshape(
                    new_out_shapes[idx], order='C') for idx in range(self.n_exprs)
            ]
        elif self.order == 'F':
            out = out.reshape((self.tot_out_size, nbroadcast), order='F')
            result = [
                out[self.accum_out_sizes[idx]:self.accum_out_sizes[idx+1], :].reshape(
                    new_out_shapes[idx], order='F') for idx in range(self.n_exprs)
            ]
        if self.n_exprs == 1:
            return result[0]
        else:
            return result


cdef double _scipy_callback_lambda_real(int n, double *x, void *user_data) nogil:
    cdef symengine.LambdaRealDoubleVisitor* lamb = <symengine.LambdaRealDoubleVisitor *>user_data
    cdef double result
    deref(lamb).call(&result, x)
    return result

cdef void _ctypes_callback_lambda_real(double *output, const double *input, void *user_data) nogil:
    cdef symengine.LambdaRealDoubleVisitor* lamb = <symengine.LambdaRealDoubleVisitor *>user_data
    deref(lamb).call(output, input)

IF HAVE_SYMENGINE_LLVM:
    cdef double _scipy_callback_llvm_real(int n, double *x, void *user_data) nogil:
        cdef symengine.LLVMDoubleVisitor* lamb = <symengine.LLVMDoubleVisitor *>user_data
        cdef double result
        deref(lamb).call(&result, x)
        return result

    cdef void _ctypes_callback_llvm_real(double *output, const double *input, void *user_data) nogil:
        cdef symengine.LLVMDoubleVisitor* lamb = <symengine.LLVMDoubleVisitor *>user_data
        deref(lamb).call(output, input)


def create_low_level_callable(lambdify, *args):
    from scipy import LowLevelCallable
    class LambdifyLowLevelCallable(LowLevelCallable):
        def __init__(self, lambdify, *args):
            self.lambdify = lambdify
        def __new__(cls, value, *args, **kwargs):
            return super(LambdifyLowLevelCallable, cls).__new__(cls, *args)
    return LambdifyLowLevelCallable(lambdify, *args)


cdef class LambdaDouble(_Lambdify):
    def __cinit__(self, args, *exprs, cppbool real=True, order='C', cppbool cse=False, cppbool _load=False, dtype=None):
        # reject additional arguments
        pass

    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse):
        self.lambda_double.resize(1)
        self.lambda_double[0].init(args_, outs_, cse)

    cpdef unsafe_real(self, double[::1] inp, double[::1] out, int inp_offset=0, int out_offset=0):
        self.lambda_double[0].call(&out[out_offset], &inp[inp_offset])

    cpdef unsafe_eval(self, inp, out, unsigned nbroadcast=1):
        cdef double[::1] c_inp, c_out
        cdef unsigned idx
        c_inp = np.ascontiguousarray(inp.ravel(order=self.order), dtype=self.numpy_dtype)
        c_out = out
        for idx in range(nbroadcast):
            self.lambda_double[0].call(&c_out[idx*self.tot_out_size], &c_inp[idx*self.args_size]) 

    cpdef as_scipy_low_level_callable(self):
        from ctypes import c_double, c_void_p, c_int, cast, POINTER, CFUNCTYPE
        if self.tot_out_size > 1:
            raise RuntimeError("SciPy LowLevelCallable supports only functions with 1 output")
        addr1 = cast(<size_t>&_scipy_callback_lambda_real,
                        CFUNCTYPE(c_double, c_int, POINTER(c_double), c_void_p))
        addr2 = cast(<size_t>&self.lambda_double[0], c_void_p)
        return create_low_level_callable(self, addr1, addr2)

    cpdef as_ctypes(self):
        """
        Returns a tuple with first element being a ctypes function with signature

            void func(double \*output, const double \*input, void \*user_data)

        and second element being a ctypes void pointer. This void pointer needs to be
        passed as input to the function as the third argument `user_data`.
        """
        from ctypes import c_double, c_void_p, c_int, cast, POINTER, CFUNCTYPE
        addr1 = cast(<size_t>&_ctypes_callback_lambda_real,
                        CFUNCTYPE(c_void_p, POINTER(c_double), POINTER(c_double), c_void_p))
        addr2 = cast(<size_t>&self.lambda_double[0], c_void_p)
        return addr1, addr2


cdef class LambdaComplexDouble(_Lambdify):
    def __cinit__(self, args, *exprs, cppbool real=True, order='C', cppbool cse=False, cppbool _load=False, dtype=None):
        # reject additional arguments
        pass

    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse):
        self.lambda_double.resize(1)
        self.lambda_double[0].init(args_, outs_, cse)

    cpdef unsafe_complex(self, double complex[::1] inp, double complex[::1] out, int inp_offset=0, int out_offset=0):
        self.lambda_double[0].call(&out[out_offset], &inp[inp_offset])

    cpdef unsafe_eval(self, inp, out, unsigned nbroadcast=1):
        cdef double complex[::1] c_inp, c_out
        cdef unsigned idx
        c_inp = np.ascontiguousarray(inp.ravel(order=self.order), dtype=self.numpy_dtype)
        c_out = out
        for idx in range(nbroadcast):
            self.lambda_double[0].call(&c_out[idx*self.tot_out_size], &c_inp[idx*self.args_size])


IF HAVE_SYMENGINE_LLVM:
    cdef class LLVMDouble(_LLVMLambdify):
        def __cinit__(self, args, *exprs, cppbool real=True, order='C', cppbool cse=False, cppbool _load=False, opt_level=3, dtype=None):
            self.opt_level = opt_level

        cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse):
            self.lambda_double.resize(1)
            self.lambda_double[0].init(args_, outs_, cse, self.opt_level)

        cdef _load(self, const string &s):
            self.lambda_double.resize(1)
            self.lambda_double[0].loads(s)

        def __reduce__(self):
            """
            Interface for pickle. Note that the resulting object is platform dependent.
            """
            cdef bytes s = self.lambda_double[0].dumps()
            return llvm_loading_func, (self.args_size, self.tot_out_size, self.out_shapes, self.real, \
                self.n_exprs, self.order, self.accum_out_sizes, self.numpy_dtype, s)

        cpdef unsafe_real(self, double[::1] inp, double[::1] out, int inp_offset=0, int out_offset=0):
            self.lambda_double[0].call(&out[out_offset], &inp[inp_offset])

        cpdef unsafe_eval(self, inp, out, unsigned nbroadcast=1):
            cdef double[::1] c_inp, c_out
            cdef unsigned idx
            c_inp = np.ascontiguousarray(inp.ravel(order=self.order), dtype=self.numpy_dtype)
            c_out = out
            for idx in range(nbroadcast):
                self.lambda_double[0].call(&c_out[idx*self.tot_out_size], &c_inp[idx*self.args_size])

        cpdef as_scipy_low_level_callable(self):
            from ctypes import c_double, c_void_p, c_int, cast, POINTER, CFUNCTYPE
            if not self.real:
                raise RuntimeError("Lambda function has to be real")
            if self.tot_out_size > 1:
                raise RuntimeError("SciPy LowLevelCallable supports only functions with 1 output")
            addr1 = cast(<size_t>&_scipy_callback_llvm_real,
                            CFUNCTYPE(c_double, c_int, POINTER(c_double), c_void_p))
            addr2 = cast(<size_t>&self.lambda_double[0], c_void_p)
            return create_low_level_callable(self, addr1, addr2)

        cpdef as_ctypes(self):
            """
            Returns a tuple with first element being a ctypes function with signature

                void func(double * output, const double *input, void *user_data)

            and second element being a ctypes void pointer. This void pointer needs to be
            passed as input to the function as the third argument `user_data`.
            """
            from ctypes import c_double, c_void_p, c_int, cast, POINTER, CFUNCTYPE
            if not self.real:
                raise RuntimeError("Lambda function has to be real")
            addr1 = cast(<size_t>&_ctypes_callback_llvm_real,
                            CFUNCTYPE(c_void_p, POINTER(c_double), POINTER(c_double), c_void_p))
            addr2 = cast(<size_t>&self.lambda_double[0], c_void_p)
            return addr1, addr2

    cdef class LLVMFloat(_LLVMLambdify):
        def __cinit__(self, args, *exprs, cppbool real=True, order='C', cppbool cse=False, cppbool _load=False, opt_level=3, dtype=None):
            self.opt_level = opt_level

        cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse):
            self.lambda_double.resize(1)
            self.lambda_double[0].init(args_, outs_, cse, self.opt_level)

        cdef _load(self, const string &s):
            self.lambda_double.resize(1)
            self.lambda_double[0].loads(s)

        def __reduce__(self):
            """
            Interface for pickle. Note that the resulting object is platform dependent.
            """
            cdef bytes s = self.lambda_double[0].dumps()
            return llvm_float_loading_func, (self.args_size, self.tot_out_size, self.out_shapes, self.real, \
                self.n_exprs, self.order, self.accum_out_sizes, self.numpy_dtype, s)

        cpdef unsafe_real(self, float[::1] inp, float[::1] out, int inp_offset=0, int out_offset=0):
            self.lambda_double[0].call(&out[out_offset], &inp[inp_offset])

        cpdef unsafe_eval(self, inp, out, unsigned nbroadcast=1):
            cdef float[::1] c_inp, c_out
            cdef unsigned idx
            c_inp = np.ascontiguousarray(inp.ravel(order=self.order), dtype=self.numpy_dtype)
            c_out = out
            for idx in range(nbroadcast):
                self.lambda_double[0].call(&c_out[idx*self.tot_out_size], &c_inp[idx*self.args_size])

    IF HAVE_SYMENGINE_LLVM_LONG_DOUBLE:
        cdef class LLVMLongDouble(_LLVMLambdify):
            def __cinit__(self, args, *exprs, cppbool real=True, order='C', cppbool cse=False, cppbool _load=False, opt_level=3, dtype=None):
                self.opt_level = opt_level

            cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse):
                self.lambda_double.resize(1)
                self.lambda_double[0].init(args_, outs_, cse, self.opt_level)

            cdef _load(self, const string &s):
                self.lambda_double.resize(1)
                self.lambda_double[0].loads(s)

            def __reduce__(self):
                """
                Interface for pickle. Note that the resulting object is platform dependent.
                """
                cdef bytes s = self.lambda_double[0].dumps()
                return llvm_long_double_loading_func, (self.args_size, self.tot_out_size, self.out_shapes, self.real, \
                    self.n_exprs, self.order, self.accum_out_sizes, self.numpy_dtype, s)

            cpdef unsafe_real(self, long double[::1] inp, long double[::1] out, int inp_offset=0, int out_offset=0):
                self.lambda_double[0].call(&out[out_offset], &inp[inp_offset])

            cpdef unsafe_eval(self, inp, out, unsigned nbroadcast=1):
                cdef long double[::1] c_inp, c_out
                cdef unsigned idx
                c_inp = np.ascontiguousarray(inp.ravel(order=self.order), dtype=self.numpy_dtype)
                c_out = out
                for idx in range(nbroadcast):
                    self.lambda_double[0].call(&c_out[idx*self.tot_out_size], &c_inp[idx*self.args_size])

    def llvm_loading_func(*args):
        return LLVMDouble(args, _load=True)

    def llvm_float_loading_func(*args):
        return LLVMFloat(args, _load=True)

    IF HAVE_SYMENGINE_LLVM_LONG_DOUBLE:
        def llvm_long_double_loading_func(*args):
            return LLVMLongDouble(args, _load=True)

def Lambdify(args, *exprs, cppbool real=True, backend=None, order='C',
             as_scipy=False, cse=False, dtype=None, **kwargs):
    """
    Lambdify instances are callbacks that numerically evaluate their symbolic
    expressions from user provided input (real or complex) into (possibly user
    provided) output buffers (real or complex). Multidimensional data are
    processed in their most cache-friendly way (i.e. "ravelled").

    Parameters
    ----------
    args: iterable of Symbols
    \*exprs: array_like of expressions
        the shape of exprs is preserved
    real : bool
        Whether datatype is ``double`` (``double complex`` otherwise).
    backend : str
        'llvm' or 'lambda'. When ``None`` the environment variable
        'SYMENGINE_LAMBDIFY_BACKEND' is used (taken as 'lambda' if unset).
    order : 'C' or 'F'
        C- or Fortran-contiguous memory layout. Note that this affects
        broadcasting: e.g. a (m, n) matrix taking 3 arguments and given a
        (k, l, 3) (C-contiguous) input will give a (k, l, m, n) shaped output,
        whereas a (3, k, l) (C-contiguous) input will give a (m, n, k, l) shaped
        output. If ``None`` order is taken as ``self.order`` (from initialization).
    as_scipy : bool
        return a SciPy LowLevelCallable which can be used in SciPy's integrate
        methods
    cse : bool
        Run Common Subexpression Elimination on the output before generating
        the callback.
    dtype : numpy.dtype type

    Returns
    -------
    callback instance with signature f(inp, out=None)

    Examples
    --------
    >>> from symengine import var, Lambdify
    >>> var('x y z')
    >>> f = Lambdify([x, y, z], [x+y+z, x*y*z])
    >>> f([2, 3, 4])
    [ 9., 24.]
    >>> out = np.array(2)
    >>> f(x, out); out
    [ 9., 24.]

    """
    if backend is None:
        backend = os.getenv('SYMENGINE_LAMBDIFY_BACKEND', "lambda")
    if backend == "llvm":
        IF HAVE_SYMENGINE_LLVM:
            if dtype == None:
                dtype = np.float64
            if dtype == np.float64:
                ret = LLVMDouble(args, *exprs, real=real, order=order, cse=cse, dtype=np.float64, **kwargs)
            elif dtype == np.float32:
                ret = LLVMFloat(args, *exprs, real=real, order=order, cse=cse, dtype=np.float32, **kwargs)
            elif dtype == np.longdouble:
                IF HAVE_SYMENGINE_LLVM_LONG_DOUBLE:
                    ret = LLVMLongDouble(args, *exprs, real=real, order=order, cse=cse, dtype=np.longdouble, **kwargs)
                ELSE:
                    raise ValueError("Long double not supported on this platform")
            else:
                raise ValueError("Unknown numpy dtype.")
                
            if as_scipy:
                return ret.as_scipy_low_level_callable()
            return ret
        ELSE:
            raise ValueError("""llvm backend is chosen, but symengine is not compiled
                                with llvm support.""")
    elif backend == "lambda":
        pass
    else:
        warnings.warn("Unknown SymEngine backend: %s\nUsing backend='lambda'" % backend)
    if real:
        ret = LambdaDouble(args, *exprs, real=real, order=order, cse=cse, **kwargs)
    else:
        ret = LambdaComplexDouble(args, *exprs, real=real, order=order, cse=cse, **kwargs)
    if as_scipy:
        return ret.as_scipy_low_level_callable()
    return ret


def LambdifyCSE(args, *exprs, order='C', **kwargs):
    """ Analogous with Lambdify but performs common subexpression elimination.
    """
    warnings.warn("LambdifyCSE is deprecated. Use Lambdify(..., cse=True)", DeprecationWarning)
    return Lambdify(args, *exprs, cse=True, order=order, **kwargs)


def ccode(expr):
    cdef Basic expr_ = sympify(expr)
    return symengine.ccode(deref(expr_.thisptr)).decode("utf-8")


def piecewise(*v):
    cdef symengine.PiecewiseVec vec
    cdef pair[rcp_const_basic, RCP[symengine.const_Boolean]] p
    cdef Basic e
    cdef Boolean b
    for expr, rel in v:
        e = sympify(expr)
        b = sympify(rel)
        p.first = <rcp_const_basic>e.thisptr
        p.second = <RCP[symengine.const_Boolean]>symengine.rcp_static_cast_Boolean(b.thisptr)
        vec.push_back(p)
    return c2py(symengine.piecewise(symengine.std_move_PiecewiseVec(vec)))


def interval(start, end, left_open=False, right_open=False):
    if isinstance(start, NegativeInfinity):
        left_open = True
    if isinstance(end, Infinity):
        right_open = True
    cdef Number start_ = sympify(start)
    cdef Number end_ = sympify(end)
    cdef cppbool left_open_ = left_open
    cdef cppbool right_open_ = right_open
    cdef RCP[const symengine.Number] n1 = symengine.rcp_static_cast_Number(start_.thisptr)
    cdef RCP[const symengine.Number] n2 = symengine.rcp_static_cast_Number(end_.thisptr)
    return c2py(symengine.interval(n1, n2, left_open_, right_open_))


def emptyset():
    return c2py(<rcp_const_basic>(symengine.emptyset()))


def universalset():
    return c2py(<rcp_const_basic>(symengine.universalset()))


def reals():
    return c2py(<rcp_const_basic>(symengine.reals()))


def integers():
    return c2py(<rcp_const_basic>(symengine.integers()))


def finiteset(*args):
    cdef symengine.set_basic s
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        s.insert(<rcp_const_basic>(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.finiteset(s)))


def contains(expr, sset):
    cdef Basic expr_ = sympify(expr)
    cdef Set sset_ = sympify(sset)
    cdef RCP[const symengine.Set] s = symengine.rcp_static_cast_Set(sset_.thisptr)
    return c2py(<rcp_const_basic>(symengine.contains(expr_.thisptr, s)))


def tribool(value):
    if value == -1:
        return None
    else:
        return bool(value)


def is_zero(expr):
    cdef Basic expr_ = sympify(expr)
    cdef int tbool = symengine.is_zero(deref(expr_.thisptr))
    return tribool(tbool)


def is_positive(expr):
    cdef Basic expr_ = sympify(expr)
    cdef int tbool = symengine.is_positive(deref(expr_.thisptr))
    return tribool(tbool)


def is_negative(expr):
    cdef Basic expr_ = sympify(expr)
    cdef int tbool = symengine.is_negative(deref(expr_.thisptr))
    return tribool(tbool)


def is_nonpositive(expr):
    cdef Basic expr_ = sympify(expr)
    cdef int tbool = symengine.is_nonpositive(deref(expr_.thisptr))
    return tribool(tbool)


def is_nonnegative(expr):
    cdef Basic expr_ = sympify(expr)
    cdef int tbool = symengine.is_nonnegative(deref(expr_.thisptr))
    return tribool(tbool)


def is_real(expr):
    cdef Basic expr_ = sympify(expr)
    cdef int tbool = symengine.is_real(deref(expr_.thisptr))
    return tribool(tbool)


def set_union(*args):
    cdef symengine.set_set s
    cdef Set e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Set(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.set_union(s)))


def set_intersection(*args):
    cdef symengine.set_set s
    cdef Set e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Set(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.set_intersection(s)))


def set_complement(universe, container):
    cdef Set universe_ = sympify(universe)
    cdef Set container_ = sympify(container)
    cdef RCP[const symengine.Set] u = symengine.rcp_static_cast_Set(universe_.thisptr)
    cdef RCP[const symengine.Set] c = symengine.rcp_static_cast_Set(container_.thisptr)
    return c2py(<rcp_const_basic>(symengine.set_complement(u, c)))


def set_complement_helper(container, universe):
    cdef Set container_ = sympify(container)
    cdef Set universe_ = sympify(universe)
    cdef RCP[const symengine.Set] c = symengine.rcp_static_cast_Set(container_.thisptr)
    cdef RCP[const symengine.Set] u = symengine.rcp_static_cast_Set(universe_.thisptr)
    return c2py(<rcp_const_basic>(symengine.set_complement_helper(c, u)))


def conditionset(sym, condition):
    cdef Basic sym_ = sympify(sym)
    cdef Boolean condition_ = sympify(condition)
    cdef RCP[const symengine.Boolean] c = symengine.rcp_static_cast_Boolean(condition_.thisptr)
    return c2py(<rcp_const_basic>(symengine.conditionset(sym_.thisptr, c)))


def imageset(sym, expr, base):
    cdef Basic sym_ = sympify(sym)
    cdef Basic expr_ = sympify(expr)
    cdef Set base_ = sympify(base)
    cdef RCP[const symengine.Set] b = symengine.rcp_static_cast_Set(base_.thisptr)
    return c2py(<rcp_const_basic>(symengine.imageset(sym_.thisptr, expr_.thisptr, b)))


universal_set_singleton = UniversalSet()
integers_singleton = Integers()
reals_singleton = Reals()
empty_set_singleton = EmptySet()


def solve(f, sym, domain=None):
    cdef Basic f_ = sympify(f)
    cdef Basic sym_ = sympify(sym)
    require(sym_, Symbol)
    cdef RCP[const symengine.Symbol] x = symengine.rcp_static_cast_Symbol(sym_.thisptr)
    if domain is None:
        return c2py(<rcp_const_basic>(symengine.solve(f_.thisptr, x)))
    cdef Set domain_ = sympify(domain)
    cdef RCP[const symengine.Set] d = symengine.rcp_static_cast_Set(domain_.thisptr)
    return c2py(<rcp_const_basic>(symengine.solve(f_.thisptr, x, d)))


def linsolve(eqs, syms):
    """
    Solve a set of linear equations given as an iterable `eqs`
    which are linear w.r.t the symbols given as an iterable `syms`
    """
    cdef symengine.vec_basic eqs_ = iter_to_vec_basic(eqs)
    cdef symengine.vec_sym syms_
    cdef RCP[const symengine.Symbol] sym_
    cdef Symbol B
    for sym in syms:
        B = sympify(sym)
        sym_ = symengine.rcp_static_cast_Symbol(B.thisptr)
        syms_.push_back(sym_)
    if syms_.size() != eqs_.size():
        raise RuntimeError("Number of equations and symbols do not match")
    cdef symengine.vec_basic ret = symengine.linsolve(eqs_, syms_)
    return vec_basic_to_tuple(ret)


def cse(exprs):
    cdef symengine.vec_basic vec
    cdef symengine.vec_pair replacements
    cdef symengine.vec_basic reduced_exprs
    cdef Basic b
    for expr in exprs:
        b = sympify(expr)
        vec.push_back(b.thisptr)
    symengine.cse(replacements, reduced_exprs, vec)
    return (vec_pair_to_list(replacements), vec_basic_to_list(reduced_exprs))

def latex(expr):
    cdef Basic expr_ = sympify(expr)
    return symengine.latex(deref(expr_.thisptr)).decode("utf-8")

cdef _flattened_vec(symengine.vec_basic &vec, exprs):
    cdef Basic b
    if is_sequence(exprs):
        for expr in exprs:
            _flattened_vec(vec, expr)
    else:
        b = sympify(exprs)
        vec.push_back(b.thisptr)


def count_ops(*exprs):
    cdef symengine.vec_basic vec
    _flattened_vec(vec, exprs)
    return symengine.count_ops(vec)


# Turn on nice stacktraces:
symengine.print_stack_on_segfault()
