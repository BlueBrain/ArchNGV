
from .priority_heap cimport MinPriorityHeap
from .priority_heap cimport MaxPriorityHeap
from .priority_heap cimport PriorityHeapRecord


cpdef test_queue():

    

    pmax = MaxPriorityHeap(100)
    pmin = MinPriorityHeap(100)

    pmax.push(0, 0.4)
    pmax.push(1, 11.0)
    pmax.push(2, 1.2)
    pmax.push(3, 100.)
    pmax.push(4, 51.)
    pmax.push(5, 100000.)

    pmin.push(0, 0.4)
    pmin.push(1, 11.0)
    pmin.push(2, 1.2)
    pmin.push(3, 100.)
    pmin.push(4, 51.)
    pmin.push(5, 100000.)

    cdef PriorityHeapRecord r

    print "pmax"
    print "expected 5 3 4 1 2 0"
    for _ in range(6):

        pmax.pop(&r)
        print r.node_id, r.value


    pmax.push(0, 0.4)
    pmax.push(1, 11.0)
    pmax.push(2, 1.2)
    pmax.push(3, 100.)
    pmax.push(4, 51.)
    pmax.push(5, 100000.)

    pmax.update(3, 0.0)
    pmax.update(2, 10000.)

    print "pmax updated"
    print "expected: 5 2 4 1 0 3"
    for _ in range(6):

        pmax.pop(&r)
        print r.node_id, r.value

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


