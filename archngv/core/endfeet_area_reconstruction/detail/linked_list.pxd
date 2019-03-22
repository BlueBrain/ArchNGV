import numpy as np
cimport numpy as np


cdef packed struct Node:
    unsigned long value
    Node* next


cdef class SinglyLinkedList:

    cdef Node *head, *tail
    cdef unsigned long size

    cpdef unsigned long[:] values(self)
    cpdef head_value(self)
    cpdef tail_value(self)
    cpdef void add_left(self, unsigned long value)
    cpdef void add_right(self, unsigned long value)
    cpdef print_elements_downstream(self)
