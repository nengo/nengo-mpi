import subprocess
import tempfile
import os


def write_metis_input_file(filter_graph):
    """
    Writes a filter graph (created via network_to_filter_graph) to file indices
    the format required by metis (a graph-partitioning utility).

    Parameters
    ----------
    filter_graph: networkx Graph
        A graph created from a network using network_to_filter_graph.

    Returns
    -------
    filename: string
        The name of the file stroing the graph.
    """
    f = tempfile.NamedTemporaryFile(mode='w', prefix='metis', delete=False)

    print "Writing metis file: %s" % f.name

    with f:
        m = filter_graph.number_of_edges()
        n = filter_graph.number_of_nodes()

        f.write("%d %d 011" % (n, m))

        indices = {node: i+1 for i, node in enumerate(filter_graph.nodes())}

        for u in filter_graph.nodes():
            f.write('\n')

            vertex_weight = 0

            for obj in u.objects:
                if hasattr(obj, 'n_neurons'):
                    vertex_weight += obj.n_neurons

            f.write("%d" % vertex_weight)

            for v, weight_dict in filter_graph[u].iteritems():
                f.write(" %d %d" % (indices[v], weight_dict['weight']))

    return f.name


def read_metis_output_file(filename):
    """
    Read the given file name, assuming it is the output from a run of gpmetis.
    The format of the file is: n lines (n is the number of nodes), the i-th
    line giving information about the i-th node. Each line contains a single
    int giving the component that the i-th node is assigned to. Components are
    indexed starting from 0.
    """
    print "Reading metis output file: %s" % filename

    node_assignments = []

    with open(filename, 'r') as f:
        for line in iter(f.readline, ''):
            node_assignments.append(int(line))

    return node_assignments


def metis_output_filename(filename, n_components):
    return '%s.part.%d' % (filename, n_components)


def metis_partitioner(filter_graph, n_components, delete_file=True):
    """
    Partitions a filter graph using the metis partitioning package.

    Parameters
    ----------
    filter_graph: networkx Graph
        A graph created from a network using network_to_filter_graph.

    n_components: int
        Desired number of components in the partition.

    Returns
    -------
    assignments: dict
        A mapping from nodes in the filter graph to components.
    """

    assert n_components > 1

    filename = write_metis_input_file(filter_graph)

    print "Running metis..."
    # run metis on written file
    subprocess.check_call(['gpmetis', filename, str(n_components)])

    output_filename = metis_output_filename(filename, n_components)
    node_assignments = read_metis_output_file(output_filename)

    if delete_file:
        os.remove(filename)

    return {
        node: component
        for node, component
        in zip(filter_graph.nodes(), node_assignments)}
