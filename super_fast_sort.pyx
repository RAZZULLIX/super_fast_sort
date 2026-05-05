# super_fast_sort.pyx
# Zero-overhead Cython wrapper for your ultra-fast sorter

cimport numpy as cnp
import numpy as np

from libc.stdint cimport uint32_t
from libc.stddef cimport size_t

# Declare the functions WITHOUT including the whole file again
cdef extern from "program.c":
    void super_fast_sort_uint32(uint32_t* arr, size_t n) nogil
    void cleanup_sort_buffers() nogil


cpdef void super_fast_sort_inplace(cnp.uint32_t[::1] arr) noexcept nogil:
    """Internal fast in-place sort"""
    if arr.shape[0] <= 1:
        return
    super_fast_sort_uint32(&arr[0], arr.shape[0])


def super_fast_sort(arr):
    """
    Public API - sorts numpy uint32 array in-place.
    Returns the array (now sorted) for compatibility with yelodetermine.py
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if arr.dtype != np.uint32:
        arr = np.asarray(arr, dtype=np.uint32, copy=False)

    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.uint32)

    cdef cnp.uint32_t[::1] mv = arr

    with nogil:
        super_fast_sort_inplace(mv)

    return arr


# super_fast_sort.pyx — rename Python wrapper
def cleanup():                    # was def cleanup_sort_buffers()
    """Free internal thread-local buffers"""
    cleanup_sort_buffers()        # now unambiguously the C extern