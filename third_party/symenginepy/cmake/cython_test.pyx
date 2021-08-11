# Test that numpy works in Cython:
from numpy cimport ndarray

# Test that libcpp module is present:
from libcpp.vector cimport vector
from libcpp.string cimport string

# Test the math library:
from libc.math cimport sin, cos, atan2

# Test that templates work:
cdef vector[int] array2vector_int(a):
    cdef vector[int] v
    for i in range(len(a)):
        v.push_back(a[i])
    return v

# Test that bool support works
from libcpp cimport bool

# Test that C++ support works

cdef extern from "electrostatics.h":

    ctypedef double scalar

    cdef cppclass Function:
        pass

    cdef cppclass MeshFunction(Function):
        scalar get_pt_value(double x, double y)

    cdef cppclass Solution(MeshFunction):
        void copy(Solution *s)

    cdef cppclass Electrostatics:
        void set_mesh_str(char *mesh)
        void set_initial_mesh_refinement(int init_ref_num)
        void set_initial_poly_degree(int p)
        void set_material_markers(vector[int] &mat_markers)
        void set_permittivity_array(vector[double] &p_array)
        void set_charge_density_array(vector[double] &cd_array)
        void set_boundary_markers_value(vector[int] &bdy_markers_val)
        void set_boundary_values(vector[double] &bc_val)
        void set_boundary_markers_derivative(vector[int] &bdy_markers_der)
        void set_boundary_derivatives(vector[double] &bc_der)
        bool calculate(Solution* phi)

cdef extern from "<symengine/symengine_rcp.h>" namespace "SymEngine":
    cdef enum ENull:
        null

    cdef cppclass RCP[T]:
        T& operator*() nogil except +
        void reset() nogil except +

    cdef cppclass Ptr[T]:
        T& operator*() nogil except +

    RCP[Symbol] rcp_static_cast_Symbol "SymEngine::rcp_static_cast<SymEngine::Symbol>"(const RCP[Basic] &b) nogil
    RCP[Add] rcp_static_cast_Add "SymEngine::rcp_static_cast<SymEngine::Add>"(const RCP[Basic] &b) nogil


cdef extern from "<symengine/basic.h>" namespace "SymEngine":
    cdef cppclass Basic:
        string __str__() nogil except +
        RCP[Basic] diff(const RCP[Symbol] &x) nogil except +

    bool eq(RCP[Basic] &a, RCP[Basic] &b) nogil except +
    bool neq(RCP[Basic] &a, RCP[Basic] &b) nogil except +

    bool is_a_Add "SymEngine::is_a<SymEngine::Add>"(const Basic &b) nogil
    bool is_a_Mul "SymEngine::is_a<SymEngine::Mul>"(const Basic &b) nogil

cdef extern from "<symengine/symbol.h>" namespace "SymEngine":
    cdef cppclass Symbol(Basic):
        Symbol(string name) nogil
        string get_name() nogil

cdef extern from "<symengine/add.h>" namespace "SymEngine":
    cdef RCP[Basic] add(RCP[Basic] &a, RCP[Basic] &b) nogil except+
    cdef RCP[Basic] sub(RCP[Basic] &a, RCP[Basic] &b) nogil except+

    cdef cppclass Add(Basic):
        void as_two_terms(const Ptr[RCP[Basic]] &a, const Ptr[RCP[Basic]] &b)
