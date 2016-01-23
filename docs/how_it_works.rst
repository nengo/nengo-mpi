.. _how_it_works:

How It Works
============

Here we attempt to give a rough idea of how nengo_mpi works under the hood, and, in particular, how it achieves parallelization. nengo_mpi is based heavily on the reference implementation of nengo. The reference implementation works by converting a high-level neural model specification into a low-level computation graph. The computation graph is a collection of ``operators`` and ``signals``. In short, signals store data, and operators perform computation on signals and store the results in other signals. To run the simulation, nengo simply executes each operator in the computation graph once per time step. For a concrete example of how this works, consider the following simple nengo script: ::

    import nengo

    model = nengo.Network()
    with model:
        A = nengo.Ensemble(n_neurons=50, dimensions=1)
        B = nengo.Ensemble(n_neurons=50, dimensions=1)
        conn = nengo.Connection(A, B)

    sim = nengo.Simulator(model)
    sim.run(time_in_seconds=1.0)

The conversion from the high-level specification (e.g. the nengo Network stored in the variable ``model``)  to computation graph is called the **build** step, and takes place in the line ``sim = nengo.Simulator(model)``. The generated computation graph looks something like this:

.. image :: images/nengo_signals.svg

A few signals and operators whose purposes are somewhat opaque have been omitted here for clarity. Now suppose that we're impatient and find that the call to ``sim.run`` is too slow. We can easily parallelize the simulation step by making use of nengo_mpi. Making the few necessary changes, we end up with the following script: ::

    import nengo
    import nengo_mpi

    model = nengo.Network()
    with model:
        A = nengo.Ensemble(n_neurons=50, dimensions=1)
        B = nengo.Ensemble(n_neurons=50, dimensions=1)
        nengo.Connection(A, B)

    # assign the ensembles to different processors
    assignments = {A: 0, B: 1}
    sim = nengo_mpi.Simulator(model, assignments=assignments)

    sim.run(time_in_seconds=1.0)

Now ensembles A and B will be simulated on different processors, and we should get a factor of 2 speedup in running the simulation (though it will hardly be perceptible given how tiny our network is). nengo_mpi will produce a computation graph quite similar to the one produced by vanilla nengo, except it will use operators that are implemented in C++ rather than python, and will add a few new operators to achieve the inter-process communication:

.. image :: images/nengo_mpi_signals.svg

The ``MPISend`` operator stores the index of the processor to send its data to,
and likewise the ``MPIRecv`` operator stores the index of the processor to receive data from.
Moreover, they both share a "tag", a unique identifier which bonds the two
operators together and ensures that the data from the ``MPISend`` operator gets
sent to the correct ``MPIRecv`` operator. This basic pattern can be scaled up to
simulate very large networks on thousands of processors.

Some readers may have noticed something odd by now: it may seem like it would
be impossible to achieve accelerated performance from the set-up depicted in
the above diagrams. In particular, it seems as if the operators on processor
1 will need to wait for the results from processor 0, so the computation is
still ultimately a serial one, just that now we have added inter-process
communication in the pipeline to slow things down.

This turns out not to be the case, because the ``Synapse`` operator is special
in that it is what we call an "update" operator. Update operators break the computation
graph up into independently-simulatable components. In the first diagram, the
``DotInc`` operator in ensemble B performs computation on the value of the ``Input``
signal *from the previous time-step* [#]_. Thus, the operators in ensemble B do not need to
wait for the operators in ensemble A and the connection, since the values from the
previous time-step should already be available. Likewise, in the second diagram,
the ``MPIRecv`` operator actually receives data from the previous time-step.
Thanks to this mechanism, we are in fact able to achieve large-scale parallelization,
demonstrated empirically by our :ref:`benchmarks`.

.. [#] "Delays" like this are necessary from a biological-plausibility standpoint as well. Otherwise, neural activity elicited by a stimulus could be propogated throughout the entire network in a single time step, regardless of the network's size.