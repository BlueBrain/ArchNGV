cimport numpy as np


ctypedef np.npy_float32 DTYPE_t # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_uintp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (SIZE_t**)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (PriorityHeapRecord*)


cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *


cdef struct PriorityHeapRecord:
    SIZE_t node_id
    float value


cdef class MinPriorityHeap:

    cdef SIZE_t capacity
    cdef SIZE_t heap_ptr
    cdef PriorityHeapRecord* heap_
    cdef SIZE_t* idmap_

    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t node_id, float value) nogil except -1
    cdef int pop(self, PriorityHeapRecord* res) nogil
    cdef void heapify_up(self, PriorityHeapRecord* heap, SIZE_t* idmap, SIZE_t pos) nogil
    cdef void heapify_down(self, PriorityHeapRecord* heap, SIZE_t* idmap, SIZE_t pos, SIZE_t heap_length) nogil
    cdef int update(self, SIZE_t node_id, float new_value) nogil except -1
