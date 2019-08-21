from .priority_heap cimport MinPriorityHeap
from .priority_heap cimport PriorityHeapRecord


cpdef test_queue():

    pmin = MinPriorityHeap(100)

    pmin.push(0, 0.4)
    pmin.push(1, 11.0)
    pmin.push(2, 1.2)
    pmin.push(3, 100.)
    pmin.push(4, 51.)
    pmin.push(5, 100000.)

    cdef PriorityHeapRecord r

    print "pmin"
    print "expected: 0 2 1 4 3 5"
    for _ in range(6):

        pmin.pop(&r)
        print r.node_id, r.value

    pmin.push(0, 0.4)
    pmin.push(1, 11.0)
    pmin.push(2, 1.2)
    pmin.push(3, 100.)
    pmin.push(4, 51.)
    pmin.push(5, 100000.)

    pmin.update(4, 1.)
    pmin.update(0, 999999.)

    print "pmin updated"
    print "expected 4 2 1 3 5 0"
    for _ in range(6):
        pmin.pop(&r)
        print r.node_id, r.value
