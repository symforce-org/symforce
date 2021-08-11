// python header pyconfig.h defines hypot to be _hypot and therefore build might fail in MinGW
// http://stackoverflow.com/questions/10660524/error-building-boost-1-49-0-with-gcc-4-7-0
// This file checks whether the build fail

#include <Python.h>
#include <math.h>


