#include "Python.h"
#include "ufuncsmodule.h"

/* now we initialize the Python module which contains our new object: */
static PyModuleDef _ufuncs_module = {
    PyModuleDef_HEAD_INIT,
    "_ufuncs",
    "",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__ufuncs(void)
{
    PyObject* m;
    PyObject* d;

    m = PyModule_Create(&_ufuncs_module);
    if (m == NULL)
	return NULL;

    d = PyModule_GetDict(m);
    if (d == NULL) {
	Py_XDECREF(m);
	return NULL;
    }

    import_array();
    import_umath();

    if (InitOperators(d) < 0) {
	Py_XDECREF(d);
	Py_XDECREF(m);
	return NULL;
    }

    return m;
}

