
def split_items(N, S):
    '''
    Integer Spliting in S parts
    '''
    # integer division
    D = N / S
    # remainder
    R = N % S

    T = 0
    subsets = []
    while T < N:
        if R > 0:
            subsets.append((T, T + D + 1))
            T += D + 1
            R -= 1
        else:
            subsets.append((T, T + D))
            T += D

    return subsets

def multiprocessing_process(func, object_list):
    from multiprocessing import Process, cpu_count, Queue

    def return_vals(func, obj_list, n, queue):

        for k, obj in enumerate(obj_list):

            queue.put((k + n, func(obj)))

    N = len(object_list)
    S = min(cpu_count(), N)

    slices = split_items(N, S)

    queue = Queue()

    jobs = []

    s = 0

    for n, (start, end) in enumerate(slices):

        obj_slice = object_list[start: end]

        p = Process(target=return_vals,
                    args=(func, obj_slice, s, queue))

        s += len(obj_slice)

        p.start()
        jobs.append(p)

    for job in jobs:
        job.join()

    results = []
    # get all the elements from the queue
    while not queue.empty():

        results.append(queue.get())

    # sort with respect to the element sequence
    results = sorted(results, key=lambda el: el[0])

    # remove the indices
    results = [el[1] for el in results]

    return results


def distribute_batches(func, pool_map, object_list):

    N = len(object_list)
    S = min(pool_map._processes, N)

    slices = split_items(N, S)

    obj_gen = (obj_list[start: end] for (start, end) in slices)

    return [el for batch in pool_map(func, obj_gen) for el in batch]

