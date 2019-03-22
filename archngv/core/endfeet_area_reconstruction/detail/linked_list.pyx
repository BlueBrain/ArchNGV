# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from cython.view cimport array as cvarray

cdef inline void print_downstream(Node* node):

    while node != NULL:

        print node.value
        node = node.next


cdef inline Node* create_new_node(unsigned long value, Node* next_node) nogil:

    cdef Node* new_node = <Node*>malloc(sizeof(Node))

    new_node.value = value
    new_node.next = next_node

    return new_node

cdef class SinglyLinkedList:

    def __cinit__(self):

        head, tail = NULL, NULL
        size = 0

    def __dealloc__(self):

        cdef Node* current = NULL
        self.tail = NULL

        if self.head != NULL:
            while self.head != NULL:

                current = self.head

                if current.next != NULL:
                    self.head = self.head.next
                else:
                    self.head = NULL

                free(current)
                self.size -= 1

    cpdef unsigned long[:] values(self):

        cdef unsigned long[:] vals = cvarray(shape=(self.size, ), itemsize=sizeof(unsigned long), format="L")

        cdef Node* node = self.head

        cdef size_t n = 0

        while node != NULL:

            vals[n] = node.value
            node = node.next

            n += 1

        return vals


    cpdef head_value(self):
        return self.head.value

    cpdef tail_value(self):
        return self.tail.value


    cpdef void add_left(self, unsigned long value):

        cdef Node* new_node = create_new_node(value, NULL)

        if self.head == NULL:
             self.head = new_node
             self.tail = new_node
        else:

            new_node.next = self.head
            self.head = new_node

        self.size += 1

    cpdef void add_right(self, unsigned long value):

        cdef Node* new_node = create_new_node(value, NULL)

        if self.tail == NULL:
             self.tail = new_node
             self.head = new_node

        else:

            self.tail.next = new_node
            self.tail = new_node

        self.size += 1


    cpdef print_elements_downstream(self):
        print_downstream(self.head)

        print "head", self.head.value
        print "head_next", self.head.next.value
        print "tail", self.tail.value


