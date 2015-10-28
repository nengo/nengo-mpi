from heapq import heapify, heappush, heappop


def work_balanced_partitioner(cluster_graph, n_components):
    """
    Tries to give each component of the partition an equal number of
    neurons, making no attempt to minimize the weight of edges that
    straddle component boundaries.

    Parameters
    ----------
    cluster_graph: networkx Graph
        A graph created from a network using network_to_cluster_graph.

    n_components: int
        Desired number of components in the partition.

    Returns
    -------
    assignments: dict
        A mapping from nodes in the cluster graph to components.
    """
    assert n_components > 1

    components, _ = greedy_balanced_partition(
        cluster_graph.nodes(), n_components,
        key=lambda n: n.n_neurons)

    assignments = {}
    for i, c in enumerate(components):
        for n in c:
            assignments[n] = i

    return assignments


class PriorityDict(dict):
    """
    Retrieved from: http://code.activestate.com/recipes/
        522995-priority-dict-a-priority-queue-with-updatable-prio/

    Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super(PriorityDict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.iteritems()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super(PriorityDict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.
        super(PriorityDict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """
        while self:
            yield self.pop_smallest()


def greedy_balanced_partition(S, k, key=None):
    """
    Greedy algorithm for the k-part balanced partition problem.
    The problem is: Given a list of integers, and an integer k,
    group the integers into k sets such the sums of the groups are
    as close to one another as possible.

    Parameters
    ----------
    S: list
        A list of the objects to partition.

    k: int
        Desired number of components in the partition.

    key: callable
        A function of one argument which maps objects in S to numerical
        values. If S does not contain numerical values, then this function
        must be supplied.

    Returns
    -------
    components: list
        A list containing k lists. Each object from S appears in exactly one
        of the k lists.

    sizes: list
        A list of numbers, the ith item giving the size of the ith component.
    """

    if key is None:
        key = lambda x: x

    components = [set() for i in range(k)]
    sizes = [0] * k

    pq = PriorityDict()

    pq.update({i: 0 for i in range(k)})

    for n in sorted(S, reverse=True, key=key):

        smallest = pq.pop_smallest()

        components[smallest].add(n)
        sizes[smallest] += key(n)
        pq[smallest] = sizes[smallest]

    return components, sizes
