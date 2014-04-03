#simulator.py

from nengo import builder
import nengo_mpi

class Simulator(object):
    """MPI simulator for models."""

    def __init__(self, model, dt=0.001, seed=None, builder=Builder()):
        # Call the builder to build the model
        self.model = builder(model, dt)
        self.dt = dt

        # Use model seed as simulator seed if the seed is not provided
        # Note: seed is not used right now, but one day...
        self.seed = self.model.seed if seed is None else seed

        # -- map from Signal.base -> ndarray
        self.signals = SignalDict(__time__=np.asarray(0.0, dtype=np.float64))
        for op in self.model.operators:
            op.init_signals(self.signals, self.dt)

        self.dg = operator_depencency_graph(self.model.operators)
        self._step_order = [node for node in toposort(self.dg)
                            if hasattr(node, 'make_step')]
        
        self._init_mpi()

    def _init_mpi(self):

        self.mpi_sim = nengo_mpi.PythonMpiSimulatorChunk()
        #self.mpi_sim = nengo_mpi.PythonMpiSimulatorChunk(self.dt)

        for sig, numpy_array in self.signals.items():
            self.mpi_sim.add_signal(id(sig), numpy_array)

        for op in self._step_order:
            op_type = type(op)

            if op_type == builder.Reset:
                self.mpi_sim.create_Reset(id(op.dst), op.value)

            elif op_type == builder.Copy:
                self.mpi_sim.create_Copy(id(op.dst), id(op.src))

            elif op_type == builder.DotInc:
                self.mpi_sim.create_DotInc(id(op.A), id(op.X), id(op.Y))

            elif op_type == builder.ProdUpdate:
                self.mpi_sim.create_ProdUpdate(id(op.A), id(op.X), id(op.B), id(op.Y))

            elif op_type == builder.SimLIF:
                n_neurons = op.nl.n_neurons
                tau_ref = op.nl.tau_ref
                tau_rc = op.nl.tau_rc
                self.mpi_sim.create_SimLIF(n_neurons, 
                        tau_rc, tau_ref, self.dt, id(op.J), id(op.output))

            elif op_type == builder.SimLIFRate:
                n_neurons = op.nl.n_neurons
                tau_ref = op.nl.tau_ref
                tau_rc = op.nl.tau_rc
                self.mpi_sim.create_SimLIFRate(n_neurons, 
                        tau_rc, tau_ref, self.dt, id(op.J), id(op.output))

            else:
                raise NotImplementedError('NengoMPI cannot handle this operator')

    def step(self):
        """Advance the simulator by `self.dt` seconds.
        """
        self.mpi_sim.run_n_steps(1)

        self.n_steps += 1

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        steps = int(np.round(float(time_in_seconds) / self.dt))
        logger.debug("Running %s for %f seconds, or %d steps",
                     self.model.label, time_in_seconds, steps)
        self.run_steps(steps)

    def run_steps(self, steps):
        """Simulate for the given number of `dt` steps."""
        self.mpi_sim.run_n_steps(1)

        self.n_steps += steps

#    def trange(self, dt=None):
#        dt = self.dt if dt is None else dt
#        last_t = self.signals['__time__'] - self.dt
#        n_steps = self.n_steps if dt is None else int(
#            self.n_steps / (dt / self.dt))
#        return np.linspace(0, last_t, n_steps)
