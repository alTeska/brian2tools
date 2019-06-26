from numpy import shape
from brian2 import NeuronGroup, TimedArray, Equations, get_device
from .simulation import RuntimeSimulation, CPPStandaloneSimulation


def generate_fits(model, method, params, input, input_var, output_var, dt, param_init=None):
    """Generate instance of best fits for all of the traces"""

    # Check initialization of params
    for param, val in param_init.items():
        if not (param in model.identifiers or param in model.names):
            raise Exception("%s is not a model variable or an identifier in the model")

    param_names = model.parameter_names

    # set up simulator
    simulators = {
        'CPPStandaloneDevice': CPPStandaloneSimulation(),
        'RuntimeDevice': RuntimeSimulation()
    }
    simulator = simulators[get_device().__class__.__name__]

    Ntraces, Nsteps = input.shape
    duration = Nsteps * dt

    input_traces = TimedArray(input.transpose(), dt=dt)
    input_unit = input.dim
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                              ' + "% s" % repr(input_unit))

    neurons = NeuronGroup(Ntraces, model, method=method)
    neurons.namespace['input_var'] = input_traces
    neurons.namespace['Ntraces'] = Ntraces

    # initalize the values
    neurons.set_states(param_init)

    simulator.initialize_simulation(neurons)

    monitor = simulator.run_simulation(neurons, duration, params, param_names,
                                       [output_var])

    fits = getattr(monitor, output_var)

    return fits
