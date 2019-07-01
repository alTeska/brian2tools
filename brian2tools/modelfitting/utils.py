from brian2 import (NeuronGroup, TimedArray, Equations, get_device, Network,
                    StateMonitor, device)
from .simulation import RuntimeSimulation, CPPStandaloneSimulation


# TODO: get generate fits to work with standalone
def generate_fits(model, method, params, input, input_var, output_var, dt, param_init=None):
    """Generate instance of best fits for all of the traces"""

    # Check initialization of params
    for param, val in param_init.items():
        if not (param in model.identifiers or param in model.names):
            raise Exception("%s is not a model variable or an identifier in\
                            the model")

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

    neurons = NeuronGroup(Ntraces, model, method=method, name='neurons')
    neurons.namespace['input_var'] = input_traces
    neurons.namespace['Ntraces'] = Ntraces

    # initalize the values
    neurons.set_states(param_init)

    monitor = StateMonitor(neurons, output_var, record=True, name='monitor')

    network = Network()
    network.add(neurons, monitor)
    simulator.initialize(network)

    simulator.run(duration, params, param_names)

    fits = getattr(monitor, output_var)

    return fits
