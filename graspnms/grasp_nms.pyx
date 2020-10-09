# distutils: language = c++
# // Code Written by Minghao Gou
import numpy as np
cimport numpy as np
from libc.string cimport memcpy
from grasp_nms cimport *
import math

cdef double_array np2array(np.ndarray ary):
    cdef np.ndarray np_buff = np.ascontiguousarray(ary.copy(), dtype=np.float64)
    cdef double* im_buff = <double*> np_buff.data
    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    cdef double_array da = double_array(r,c)
    memcpy(da.data,im_buff,r * c * 8)
    return da

def arg_sort_grasp(np.ndarray grasps):
    cdef int num_grasp = grasps.shape[0]
    cdef np.ndarray grasp_score  = grasps[:,1]
    cdef np.ndarray sorted_arg = np.argsort(grasp_score)
    return sorted_arg

def print_data(ary):
    a = np2array(ary)
    a.print_data()

def nms_grasp(np.ndarray grasps,double t,double r):
    a = np2array(grasps)
    return array2np(grasp_nms(a,tuple_thresh(t,r)))

# cdef tuple_thresh double2tuplethresh(double t,double r):
#     return tuple_thresh(t,r)

# cdef object tuplethresh2double(tuple_thresh t):
#     return (t.translation_thresh,t.rotation_thresh)

cdef object array2np(double_array da):
    # Create buffer to transfer data from m.data
    cdef Py_buffer buf_info
    # Define the size / len of data
    cdef size_t len = da.r*da.c*sizeof(double)
    # Fill buffer
    PyBuffer_FillInfo(&buf_info, NULL, da.data, len, 1, PyBUF_FULL_RO)
    # Get Pyobject from buffer data
    Pydata  = PyMemoryView_FromBuffer(&buf_info)

    # Create ndarray with data
    shape_array = (da.r, da.c)
    ary = np.ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=np.float64)

    # BGR -> RGB
    if ary.ndim == 3 and ary.shape[2] == 3:
        ary = np.dstack((ary[...,2], ary[...,1], ary[...,0]))
    # Convert to numpy array
    pyarr = np.asarray(ary)
    if pyarr.ndim == 3 and pyarr.shape[2] == 1:
        pyarr = np.reshape(pyarr, pyarr.shape[:-1])
    return pyarr