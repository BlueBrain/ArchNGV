# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport free
from libc.stdlib cimport realloc


cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.

    cdef size_t nbytes = nelems * sizeof(p[0][0])

    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))

    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)

    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience


cdef class MaxPriorityHeap:
    """A priority queue implemented as a binary heap.
    The heap invariant is that the impurity value of the parent record
    is larger then the impurity value of the children.
    Attributes
    ----------
    capacity : SIZE_t
        The capacity of the heap
    heap_ptr : SIZE_t
        The water mark of the heap; the heap grows from left to right in the
        array ``heap_``. The following invariant holds ``heap_ptr < capacity``.
    heap_ : PriorityHeapRecord*
        The array of heap records. The maximum element is on the left;
        the heap grows from left to right
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.heap_ptr = 0

        safe_realloc(&self.heap_, capacity)
        safe_realloc(&self.idmap_, capacity)

    def __dealloc__(self):
        free(self.heap_)
        free(self.idmap_)

    cdef bint is_empty(self) nogil:
        return self.heap_ptr <= 0

    cdef int push(self, SIZE_t node_id, float value) nogil except -1:
        """Push record on the priority heap.
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        cdef SIZE_t heap_ptr = self.heap_ptr
        cdef PriorityHeapRecord* heap = NULL
        cdef SIZE_t* idmap = NULL

        if node_id >= self.capacity:
            with gil:
                raise IndexError("Node id is bigger than the maximum size of the allocated capacity (%d, %d)"
                                  % (node_id, self.capacity))

        # Resize if capacity not sufficient
        if heap_ptr >= self.capacity:

            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.heap_, self.capacity)
            safe_realloc(&self.idmap_, self.capacity)

        # Put element as last element of heap
        heap = self.heap_
        heap[heap_ptr].node_id = node_id
        heap[heap_ptr].value = value

        # store the current position of the id
        idmap = self.idmap_
        idmap[node_id] = heap_ptr

        # Heapify up
        self.heapify_up(heap, idmap, heap_ptr)

        # Increase element count
        self.heap_ptr = heap_ptr + 1
        return 0

    cdef int pop(self, PriorityHeapRecord* res) nogil:
        """Remove min element from the heap. """

        cdef SIZE_t heap_ptr = self.heap_ptr
        cdef PriorityHeapRecord* heap = self.heap_
        cdef SIZE_t* idmap = self.idmap_

        if heap_ptr <= 0:
            return -1

        # Take first element
        res[0] = heap[0]

        # swap with last element

        idmap[heap[0].node_id], idmap[heap[heap_ptr - 1].node_id] = idmap[heap[heap_ptr - 1].node_id], idmap[heap[0].node_id]

        heap[0], heap[heap_ptr - 1] = heap[heap_ptr - 1], heap[0]

        if heap_ptr > 1:
            self.heapify_down(heap, idmap, 0, heap_ptr - 1)


        # reduce the array length
        self.heap_ptr = heap_ptr - 1

        return 0

    cdef void heapify_up(self,
                         PriorityHeapRecord* heap,
                         SIZE_t* idmap,
                         SIZE_t pos) nogil:
        """Restore heap invariant parent.value > child.value from
           ``pos`` upwards. """
        if pos == 0:
            return

        cdef SIZE_t parent_pos = (pos - 1) / 2

        if heap[parent_pos].value < heap[pos].value:

            # swap map id pos
            idmap[heap[parent_pos].node_id], idmap[heap[pos].node_id] = \
            idmap[heap[pos].node_id], idmap[heap[parent_pos].node_id]

            heap[parent_pos], heap[pos] = heap[pos], heap[parent_pos]

            self.heapify_up(heap, idmap, parent_pos)

    cdef void heapify_down(self,
                           PriorityHeapRecord* heap,
                           SIZE_t* idmap,
                           SIZE_t pos,
                           SIZE_t heap_length) nogil:
        """Restore heap invariant parent.value > children.value from
           ``pos`` downwards. """
        cdef SIZE_t left_pos = 2 * (pos + 1) - 1
        cdef SIZE_t right_pos = 2 * (pos + 1)
        cdef SIZE_t largest = pos

        if (left_pos < heap_length and
                heap[left_pos].value > heap[largest].value):
            largest = left_pos

        if (right_pos < heap_length and
                heap[right_pos].value > heap[largest].value):
            largest = right_pos

        if largest != pos:

            # swap map id pos
            idmap[heap[pos].node_id], idmap[heap[largest].node_id] = \
            idmap[heap[largest].node_id], idmap[heap[pos].node_id]

            heap[pos], heap[largest] = heap[largest], heap[pos]

            self.heapify_down(heap, idmap, largest, heap_length)

    cdef int update(self, SIZE_t node_id, float new_value) nogil except -1:

        cdef PriorityHeapRecord* heap = self.heap_
        cdef SIZE_t* idmap = self.idmap_

        cdef SIZE_t heap_ptr = self.heap_ptr

        cdef SIZE_t pos = idmap[node_id]
        cdef float old_value = heap[pos].value


        if node_id >= self.capacity or node_id < 0:
            with gil:
                raise IndexError("Invalid Node id (%d)"
                                  % (node_id))

        # do nothing
        if old_value == new_value:
            return 0

        heap[pos].value = new_value

        if new_value > old_value:
            self.heapify_up(heap, idmap, pos)
        else:
            self.heapify_down(heap, idmap, pos, heap_ptr)

        return 0

cdef class MinPriorityHeap:

    """A priority queue implemented as a binary heap.
    The heap invariant is that the impurity value of the parent record
    is larger then the impurity value of the children.
    Attributes
    ----------
    capacity : SIZE_t
        The capacity of the heap
    heap_ptr : SIZE_t
        The water mark of the heap; the heap grows from left to right in the
        array ``heap_``. The following invariant holds ``heap_ptr < capacity``.
    heap_ : PriorityHeapRecord*
        The array of heap records. The maximum element is on the left;
        the heap grows from left to right
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.heap_ptr = 0

        safe_realloc(&self.heap_, capacity)
        safe_realloc(&self.idmap_, capacity)

    def __dealloc__(self):
        free(self.heap_)
        free(self.idmap_)

    cdef bint is_empty(self) nogil:
        return self.heap_ptr <= 0

    cdef int push(self, SIZE_t node_id, float value) nogil except -1:
        """Push record on the priority heap.
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """

        cdef SIZE_t heap_ptr = self.heap_ptr
        cdef PriorityHeapRecord* heap = NULL
        cdef SIZE_t* idmap = NULL

        # Resize if capacity not sufficient
        if heap_ptr >= self.capacity:
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.heap_, self.capacity)
            safe_realloc(&self.idmap_, self.capacity)

        # Put element as last element of heap
        heap = self.heap_
        heap[heap_ptr].node_id = node_id
        heap[heap_ptr].value = value

        # store the current position of the id
        idmap = self.idmap_
        idmap[node_id] = heap_ptr

        # Heapify up
        self.heapify_up(heap, idmap, heap_ptr)

        # Increase element count
        self.heap_ptr = heap_ptr + 1
        return 0

    cdef int pop(self, PriorityHeapRecord* res) nogil:
        """Remove min element from the heap. """

        cdef SIZE_t heap_ptr = self.heap_ptr
        cdef PriorityHeapRecord* heap = self.heap_
        cdef SIZE_t* idmap = self.idmap_

        if heap_ptr <= 0:
            return -1

        # Take first element
        res[0] = heap[0]

        # swap id pos
        idmap[heap[0].node_id], idmap[heap[heap_ptr - 1].node_id] = \
        idmap[heap[heap_ptr - 1].node_id], idmap[heap[0].node_id]

        # swap with last element
        heap[0], heap[heap_ptr - 1] = heap[heap_ptr - 1], heap[0]

        if heap_ptr > 1:
            self.heapify_down(heap, idmap, 0, heap_ptr - 1)

        # reduce the array length
        self.heap_ptr = heap_ptr - 1

        return 0

    cdef void heapify_up(self, PriorityHeapRecord* heap, SIZE_t* idmap, SIZE_t pos) nogil:
        """Restore heap invariant parent.value > child.value from
           ``pos`` upwards. """
        if pos == 0:
            return

        cdef SIZE_t parent_pos = (pos - 1) / 2

        if heap[parent_pos].value > heap[pos].value:

            # swap map id pos
            idmap[heap[parent_pos].node_id], idmap[heap[pos].node_id] = \
            idmap[heap[pos].node_id], idmap[heap[parent_pos].node_id]

            heap[parent_pos], heap[pos] = heap[pos], heap[parent_pos]
            self.heapify_up(heap, idmap, parent_pos)

    cdef void heapify_down(self, PriorityHeapRecord* heap, SIZE_t* idmap, SIZE_t pos, SIZE_t heap_length) nogil:
        """Restore heap invariant parent.value > children.value from
           ``pos`` downwards. """
        cdef SIZE_t left_pos = 2 * (pos + 1) - 1
        cdef SIZE_t right_pos = 2 * (pos + 1)
        cdef SIZE_t largest = pos

        if (left_pos < heap_length and heap[left_pos].value < heap[largest].value):
            largest = left_pos

        if (right_pos < heap_length and heap[right_pos].value < heap[largest].value):
            largest = right_pos

        if largest != pos:
            # swap map id pos
            idmap[heap[pos].node_id], idmap[heap[largest].node_id] = \
            idmap[heap[largest].node_id], idmap[heap[pos].node_id]

            heap[pos], heap[largest] = heap[largest], heap[pos]

            self.heapify_down(heap, idmap, largest, heap_length)

    cdef int update(self, SIZE_t node_id, float new_value) nogil except -1:

        cdef PriorityHeapRecord* heap = self.heap_
        cdef SIZE_t* idmap = self.idmap_

        cdef SIZE_t heap_ptr = self.heap_ptr

        cdef SIZE_t pos = idmap[node_id]
        cdef float old_value = heap[pos].value


        if node_id >= self.capacity or node_id < 0:
            with gil:
                raise IndexError("Invalid Node id (%d)"
                                  % (node_id))

        # do nothing
        if old_value == new_value:
            return 0

        heap[pos].value = new_value

        if new_value < old_value:
            self.heapify_up(heap, idmap, pos)
        else:
            self.heapify_down(heap, idmap, pos, heap_ptr)

        return 0
