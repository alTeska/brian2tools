from numpy import shape
from brian2 import NeuronGroup, TimedArray, Equations,
from .simulation import RuntimeSimulation, CPPStandaloneSimulation


def generate_fits(model, method, params, input, input_var, output_var, duration,
                  dt):
    """Generate instance of best fits for all of the traces"""

    param_names = model.parameter_names

    simulators = {
        'CPPStandaloneDevice': CPPStandaloneSimulation(),
        'RuntimeDevice': RuntimeSimulation()
    }
    simulator = simulators[get_device().__class__.__name__]

    ntraces, nsteps = shape(input)

    input_traces = TimedArray(input.transpose(), dt=dt)
    input_unit = input.dim
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                              ' + "% s" % repr(input_unit))

    neurons = NeuronGroup(ntraces, model, method=method)
    neurons.namespace['input_var'] = input_traces
    simulator.initialize_simulation(neurons)

    monitor = simulator.run_simulation(neurons, duration, params, param_names,
                                       [output_var])

    fits = getattr(monitor, output_var)

    return fits
