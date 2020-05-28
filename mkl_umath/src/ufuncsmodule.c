#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

/* now we initialize the Python module which contains our new object: */
static PyModuleDef _ufunc_module = {
    PyModuleDef_HEAD_INIT,
    "_ufunc",
    "",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__ext(void)
{
    PyObject* m;

    m = PyModule_Create(&_ufunc_module);
    if (m == NULL)
	return NULL;

    return m;
}

