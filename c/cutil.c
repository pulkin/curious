#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

static char module_name[] = "cutil";
static char module_docstring[] = "Fast routines for curious";
static char cutil_simplex_volumes_docstring[] = "Computes simplex volumes";
static char cutil_simplex_volumes_n_docstring[] = "Computes simplex neighbour volumes in embedding";
static PyObject* cutil_simplex_volumes(PyObject *dummy, PyObject *args);
static PyObject* cutil_simplex_volumes_n(PyObject *dummy, PyObject *args);

static PyMethodDef module_methods[] = {
    {"simplex_volumes", cutil_simplex_volumes, METH_VARARGS, cutil_simplex_volumes_docstring},
    {"simplex_volumes_n", cutil_simplex_volumes_n, METH_VARARGS, cutil_simplex_volumes_n_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    module_name,         /* m_name */
    module_docstring,    /* m_doc */
    -1,                  /* m_size */
    module_methods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

PyMODINIT_FUNC PyInit_cutil(void) {
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;
    import_array();
    return m;
}


static char cutil__validate(PyArrayObject *points, PyArrayObject *simplexes, npy_int embedding, PyArrayObject *neighbours_ptr, PyArrayObject *neighbours) {
    /* Validate simplex indexes */

    // Alignment
    if (PyArray_NDIM(points) != 2) {
        PyErr_Format(PyExc_ValueError, "A 2D array expected for 'points', found: %dD", PyArray_NDIM(points));
        return 0;
    }
    if (PyArray_NDIM(simplexes) != 2) {
        PyErr_Format(PyExc_ValueError, "A 2D array expected for 'simplexes', found: %dD", PyArray_NDIM(simplexes));
        return 0;
    }

    // Type
    if (PyArray_TYPE(points) != NPY_DOUBLE) {
        PyErr_Format(PyExc_ValueError, "The type of 'points' array is not double (type code: %d, expected: %d))", PyArray_TYPE(points), NPY_DOUBLE);
        return 0;
    }
    if (PyArray_TYPE(simplexes) != NPY_INT) {
        PyErr_Format(PyExc_ValueError, "The type of 'simplexes' array is not int32 (type code: %d, expected: %d))", PyArray_TYPE(simplexes), NPY_INT);
        return 0;
    }

    npy_int dims = PyArray_DIM(points, 1);
    npy_int dims_s = PyArray_DIM(simplexes, 1);
    npy_int ns = PyArray_DIM(simplexes, 0);
    npy_int np = PyArray_DIM(points, 0);

    if (dims + 1 != dims_s + embedding) {
        PyErr_Format(PyExc_ValueError, "Dimensionality mismatch: for %d-dimensional space %d-point simplexes expected (embedding: %d), found: %d",
            dims, dims + 1 - embedding, embedding, dims_s);
        return 0;
    }

    for (npy_int i=0; i<ns; i++) {
        for (npy_int j=0; j<dims_s; j++) {
            npy_int x = ((npy_int*)PyArray_GETPTR2(simplexes, i, j))[0];
            if (x<0 || x>=np) {
                PyErr_Format(PyExc_ValueError, "Point index simplexes[%d, %d] = %d is out of range %d-%d", i, j, x, 0, np-1);
                return 0;
            }
        }
    }

    if (neighbours != NULL && neighbours_ptr != NULL) {
        // Alignment
        if (PyArray_NDIM(neighbours) != 1) {
            PyErr_Format(PyExc_ValueError, "A 1D array expected for 'neighbours', found: %dD", PyArray_NDIM(neighbours));
            return 0;
        }
        if (PyArray_NDIM(neighbours_ptr) != 1) {
            PyErr_Format(PyExc_ValueError, "A 1D array expected for 'neighbours_ptr', found: %dD", PyArray_NDIM(neighbours_ptr));
            return 0;
        }
        if (PyArray_DIM(neighbours_ptr, 0) != np+1) {
            PyErr_Format(PyExc_ValueError, "The length of 'neighbours_ptr', is expected to be %d; found: %d", np+1, PyArray_DIM(neighbours_ptr, 0));
            return 0;
        }

        // Type
        if (PyArray_TYPE(neighbours) != NPY_INT) {
            PyErr_Format(PyExc_ValueError, "The type of 'neighbours' array is not int32 (type code: %d, expected: %d))", PyArray_TYPE(neighbours), NPY_INT);
            return 0;
        }
        if (PyArray_TYPE(neighbours_ptr) != NPY_INT) {
            PyErr_Format(PyExc_ValueError, "The type of 'neighbours_ptr' array is not int32 (type code: %d, expected: %d))", PyArray_TYPE(neighbours_ptr), NPY_INT);
            return 0;
        }


        npy_int nn = PyArray_DIM(neighbours, 0);

        for (npy_int i=0; i<nn; i++) {
            npy_int x = ((npy_int*)PyArray_GETPTR1(neighbours, i))[0];
            if (x<-1 || x>=np) {
                PyErr_Format(PyExc_ValueError, "Simplex index neighbours[%d] = %d is out of range %d-%d", i, x, -1, np-1);
                return 0;
            }
        }

        for (npy_int i=0; i<np+1; i++) {
            npy_int x = ((npy_int*)PyArray_GETPTR1(neighbours_ptr, i))[0];
            if (x<0 || x>nn) {
                PyErr_Format(PyExc_ValueError, "Pointer neighbours_ptr[%d] = %d is out of range %d-%d", i, x, 0, nn);
                return 0;
            }
        }
    }
    return 1;
}


static char cutil__get_dm(PyArrayObject *points, PyArrayObject *simplexes, npy_int simplex_i, npy_int point_origin_i, npy_double* output) {
    /* Write simplex into determinant matrix */
    npy_int dims = PyArray_DIM(points, 1);
    npy_int ns = PyArray_DIM(simplexes, 0);
    npy_int np = PyArray_DIM(points, 0);

    if (simplex_i < 0 || simplex_i >= ns) {
        PyErr_Format(PyExc_ValueError, "Simplex index %d is out of range %d-%d", simplex_i, 0, ns - 1);
        return 0;
    }

    if (point_origin_i < -1 || point_origin_i >= np) {
        PyErr_Format(PyExc_ValueError, "Point origin index %d is out of range %d-%d", point_origin_i, -1, np - 1);
        return 0;
    }

    if (point_origin_i == -1) point_origin_i = ((npy_int*)PyArray_GETPTR2(simplexes, simplex_i, dims))[0];
    for (npy_int j=0; j<dims; j++) {
        npy_int this_pt = ((npy_int*)PyArray_GETPTR2(simplexes, simplex_i, j))[0];
        for (npy_int k=0; k<dims; k++) {
            output[j * dims + k] = ((npy_double*)PyArray_GETPTR2(points, this_pt, k))[0] - ((npy_double*)PyArray_GETPTR2(points, point_origin_i, k))[0];
        }
    }
    return 1;
}


static npy_double cutil__det_(npy_double* points, npy_int dims, npy_int row, uint8_t* cols) {
    // Recursion exit
    if (row == dims - 1) {
        for (npy_int col=0; col<dims; col++)
            if (cols[col]) return points[dims * row + col];
        PyErr_Format(PyExc_RuntimeError, "Empty row");
        return 0;
    } else {
        npy_double result = 0;
        npy_int e = 1;
        for (npy_int col=0; col<dims; col++)
            if (cols[col]) {
                cols[col] = 0;
                result += e * cutil__det_(points, dims, row + 1, cols) * points[dims * row + col];
                e *= -1;
                cols[col] = 1;
            }
        return result / (dims - row);
    }
}


static npy_double cutil__det(npy_double* points, npy_int dims) {
    /* Volume. */
    if (dims == 1) {
        return points[0];
    }
    else if (dims == 2) {
        /* 0 1
           2 3 */
        return (points[0] * points[3] - points[1] * points[2]) / 2;
    } else if (dims == 3) {
        /* 0 1 2
           3 4 5
           6 7 8 */
        npy_double d1 = points[4] * points[8] - points[5] * points[7];
        npy_double d2 = points[3] * points[8] - points[5] * points[6];
        npy_double d3 = points[3] * points[7] - points[4] * points[6];
        return (points[0] * d1 - points[1] * d2 + points[2] * d3) / 6;
    } else {
        uint8_t cols[dims];
        memset(cols, 1, dims);
        return cutil__det_(points, dims, 0, cols);
    }
}


static npy_double cutil__volume(npy_double* points, npy_int dims) {
    return fabs(cutil__det(points, dims));
}


static PyObject* cutil_simplex_volumes(PyObject *dummy, PyObject *args) {
    /* Simplex volumes
       Args:
           points (np.ndarray): points' coordinates;
           simplexes (np.ndarray): simplexes;
       Returns:
           An array with simplex volumes. */
    PyArrayObject *points;
    PyArrayObject *simplexes;

    // Parse args
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &points, &PyArray_Type, &simplexes)) {
        PyErr_SetString(PyExc_ValueError, "Two numpy arrays are expected as arguments");
        return NULL;
    }
    if (points == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Internal error: the first argument, 'points', is NULL");
        return NULL;
    }
    if (simplexes == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Internal error: the second argument, 'simplexes', is NULL");
        return NULL;
    }

    if (!cutil__validate(points, simplexes, 0, NULL, NULL)) return NULL;

    npy_int dims = PyArray_DIM(points, 1);
    npy_int ns = PyArray_DIM(simplexes, 0);

    PyObject *output;
    npy_intp _ns_l = ns;
    output = PyArray_SimpleNew(1, &_ns_l, NPY_DOUBLE);
    if (output == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate the output array");
        return NULL;
    }

    npy_double dmatrix[dims * dims];

    for (npy_int i=0; i<ns; i++) {
        if (!cutil__get_dm(points, simplexes, i, -1, dmatrix)) return NULL;
        npy_double vol = cutil__volume(dmatrix, dims);
        if (vol < 0) return NULL;
        ((npy_double*)PyArray_GETPTR1(output, i))[0] = vol;
    }

    return output;
}


static PyObject* cutil_simplex_volumes_n(PyObject *dummy, PyObject *args) {
    /* Simplex volumes together with neighbours.
       Args:
           points (np.ndarray): points' coordinates;
           simplexes (np.ndarray): simplexes;
           neighbours_ptr (np.ndarray): simplex neighbours (ptrs);
           neighbours (np.ndarray): simplex neighbours;
       Returns:
           An array with simplex volumes. */
    PyArrayObject *points;
    PyArrayObject *simplexes;
    PyArrayObject *neighbours_ptr;
    PyArrayObject *neighbours;

    // Parse args
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &points, &PyArray_Type, &simplexes, &PyArray_Type, &neighbours_ptr, &PyArray_Type, &neighbours)) {
        PyErr_SetString(PyExc_ValueError, "Three numpy arrays are expected as arguments");
        return NULL;
    }
    if (points == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Internal error: the first argument, 'points', is NULL");
        return NULL;
    }
    if (simplexes == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Internal error: the second argument, 'simplexes', is NULL");
        return NULL;
    }
    if (neighbours_ptr == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Internal error: the third argument, 'neighbours_ptr', is NULL");
        return NULL;
    }
    if (neighbours == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Internal error: the fourth argument, 'neighbours', is NULL");
        return NULL;
    }

    if (!cutil__validate(points, simplexes, 1, neighbours_ptr, neighbours)) return NULL;

    npy_int dims = PyArray_DIM(points, 1);
    npy_int np = PyArray_DIM(points, 0);
    npy_int ns = PyArray_DIM(simplexes, 0);

    PyObject *output;
    npy_intp _ns_l = ns;
    output = PyArray_ZEROS(1, &_ns_l, NPY_DOUBLE, 0);
    if (output == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate the output array");
        return NULL;
    }

    npy_double dmatrix[dims * dims];

    for (npy_int i=0; i<ns; i++) {  // Iter over simplexes
        uint8_t neighbors_set[np];
        memset(neighbors_set, 0, sizeof(neighbors_set));
        for (npy_int j=0; j<dims; j++) {  // Iter over simplex points
            npy_int simplex_point_i = ((npy_int*)PyArray_GETPTR2(simplexes, i, j))[0];
            npy_int ptr_from = ((npy_int*)PyArray_GETPTR1(neighbours_ptr, simplex_point_i))[0];
            npy_int ptr_to = ((npy_int*)PyArray_GETPTR1(neighbours_ptr, simplex_point_i+1))[0];
            for (npy_int k=ptr_from; k<ptr_to; k++) {  // Iter over nb pts
                npy_int neighbour_point_i = ((npy_int*)PyArray_GETPTR1(neighbours, k))[0];
                if (neighbour_point_i != -1 && !neighbors_set[neighbour_point_i]) {
                    neighbors_set[neighbour_point_i] = 1;
                    if (!cutil__get_dm(points, simplexes, i, neighbour_point_i, dmatrix)) return NULL;
                    npy_double vol = cutil__volume(dmatrix, dims);
                    if (vol < 0) return NULL;
                    ((npy_double*)PyArray_GETPTR1(output, i))[0] += vol;
                }
            }
        }
    }

    return output;
}
