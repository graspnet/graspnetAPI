# distutils: language = c++
# // Code Written by Minghao Gou
cimport numpy as np
import numpy as np

cdef extern from "graspnms.cpp":
    pass

cdef extern from "graspnms.h":
    double_array grasp_nms(double_array,tuple_thresh)
    double * creat_array(int,int)
    cdef cppclass double_array:
        double_array() except +
        double_array(int,int) except +
        double* data
        int r,c
        void print_data()

    cdef cppclass tuple_thresh:
        tuple_thresh() except +
        tuple_thresh(double,double) except +
        int smaller(tuple_thresh)
        void print_thresh()
        double translation_thresh,rotation_thresh

cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int readonly, int infoflags)
    enum:
        PyBUF_FULL_RO

cdef double_array np2array(np.ndarray)
cdef object array2np(double_array da)
