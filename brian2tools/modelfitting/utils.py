from brian2 import (NeuronGroup, TimedArray, Equations, get_device, Network,
                    StateMonitor, device)
from .simulation import RuntimeSimulation, CPPStandaloneSimulation


# TODO: get generate fits to work with standalone
def generate_fits(model,
                  params,
                  input,
                  input_var,
                  output_var,
                  dt,
                  method,
                  param_init=None):
    """
    Generate instance of best fits for predicted parameters and all of the
    traces

    Parameters
    ----------
    model : `~brian2.equations.Equations` or string
        The equations describing the model.
    params : dict
        Predicted parameters
    input : input data as a 2D array
    input_var : string
        Input variable name.
    output_var : string
        Output variable name.
    dt : time step
    method: string, optional
        Integration method
    param_init: dict
        Dictionary of variables to be initialized with the value

    Returns
    -------
    fits: array
        Traces for each input current
    """

    # Check initialization of params
    if param_init:
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
    if param_init:
        neurons.set_states(param_init)

    monitor = StateMonitor(neurons, output_var, record=True, name='monitor')

    network = Network()
    network.add(neurons, monitor)
    simulator.initialize(network)

    simulator.run(duration, params, param_names)

    fits = getattr(monitor, output_var)

    return fits
