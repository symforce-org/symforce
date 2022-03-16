#ifndef SYMENGINE_PY_CALLABLE_WRAPPER_H
#define SYMENGINE_PY_CALLABLE_WRAPPER_H

#include <Python.h>
#include <string>
#include <symengine/symbol.h>

namespace SymEngine {

/**
 * This wraps a Python callable that takes no arguments and returns a string, and acts as a C++
 * functor that returns the string as a Symbol.
 *
 * This is used by CSE, which passes a python functor returning symbols as strings into C++, where
 * it needs to be passed around as a std::function and called from C++.
 *
 * Most of this class is for managing the held reference to the callable PyObject. operator()
 * contains all of the logic for calling the Python function and converting its output appropriately
 *
 * Based largely on https://stackoverflow.com/questions/39044063/pass-a-closure-from-cython-to-c
 */
class PyCallableWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyCallableWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    PyCallableWrapper(const PyCallableWrapper& rhs): PyCallableWrapper(rhs.held) {}

    PyCallableWrapper(PyCallableWrapper&& rhs): held(rhs.held) {
        rhs.held = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyCallableWrapper(): PyCallableWrapper(nullptr) {}

    ~PyCallableWrapper() {
        Py_XDECREF(held);
    }

    PyCallableWrapper& operator=(const PyCallableWrapper& rhs) {
        PyCallableWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyCallableWrapper& operator=(PyCallableWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    RCP<const Symbol> operator()() {
        if (held != NULL) {
            PyObject* args = PyTuple_New(0);
            // note, no way of checking for errors until you return to Python
            PyObject* py_result = PyObject_Call(held, args, NULL);
            Py_DECREF(args);

            PyObject* temp;
            std::string str;
#if PY_MAJOR_VERSION > 2
            temp = PyUnicode_AsUTF8String(py_result);
            str = std::string(PyBytes_AsString(temp));
#else
            temp = PyObject_Str(py_result);
            str = std::string(PyString_AsString(temp));
#endif
            Py_XDECREF(temp);
            Py_DECREF(py_result);

            return symbol(str);
        } else {
            throw std::runtime_error("Attempted to call empty PyCallableWrapper");
        }
    }

private:
    PyObject* held;
};

}

#endif //SYMENGINE_PY_CALLABLE_WRAPPER_H
