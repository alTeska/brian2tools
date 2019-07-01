from numpy import mean, ones, array, where
from brian2 import (NeuronGroup,  defaultclock, second, get_device, StateMonitor,
                    Network)
from brian2.input import TimedArray
from brian2.equations.equations import Equations
from .simulation import RuntimeSimulation, CPPStandaloneSimulation
from .metric import Metric

__all__ = ['fit_traces_standalone']


def make_dic(names, values):
    """Create dictionary based on list of strings and 2D array"""
    result_dict = dict()
    for name, value in zip(names, values):
        result_dict[name] = value

    return result_dict


def fit_traces_standalone(model=None,
                          input_var=None,
                          input=None,
                          output_var=None,
                          output=None,
                          dt=None,
                          t_start=0*second,
                          method=('linear', 'exponential_euler', 'euler'),
                          optimizer=None,
                          metric=None,
                          n_samples=10,
                          n_rounds=1,
                          verbose=True,
                          param_init=None,
                          **params):
    '''
    Creates an interface for evaluation of parameters drawn by evolutionary
    algorithms (throough ask/tell interfaces).


    Parameters
    ----------
    model : `~brian2.equations.Equations` or string
        The equations describing the model.
    input_var : string
        Input variable name.
    input : input data as a 2D array
    output_var : string
        Output variable name.
    output : output data as a 2D array
    dt : time step
    t_start: starting time of error measurement.
    method: string, optional
        Integration method
    optimizer: ~brian2tools.modelfitting.Optimizer children
        Child of Optimizer class, specific for each library.
    metric: ~brian2tools.modelfitting.Metric children
        Child of Metric class, specifies optimization metric
    n_samples: int
        Number of parameter samples to be optimized over.
    n_rounds: int
        Number of rounds to optimize over. (feedback provided over each round)
    verbose: bool
        Provide error feedback at each round
    param_init: dict
        Dictionary of variables to be initialized with the value
    **params:
        bounds for each parameter


    Returns
    -------
    result_dict : dict
        dictionary with best parameter set
    error: float
        error value for best parameter set

    TODO:
     - resolve t_start
     - tolerance

    '''

    simulators = {
        'CPPStandaloneDevice': CPPStandaloneSimulation(),
        'RuntimeDevice': RuntimeSimulation()
    }

    simulator = simulators[get_device().__class__.__name__]
    parameter_names = model.parameter_names

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')
    defaultclock.dt = dt

    # Check initialization of params
    if param_init:
        for param, val in param_init.items():
            if not (param in model.identifiers or param in model.names):
                raise Exception("%s is not a model variable or an identifier in the model")


    # Check input variable
    if input_var not in model.identifiers:
        raise Exception("%s is not an identifier in the model" % input_var)

    # Check output variable
    if output_var not in model.names:
        raise Exception("%s is not a model variable" % output_var)
    if output.shape != input.shape:
        raise Exception("Input and output must have the same size")

    Ntraces, Nsteps = input.shape
    duration = Nsteps * dt

    # Replace input variable by TimedArray
    input_traces = TimedArray(input.transpose(), dt=dt)

    input_unit = input.dim
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                              ' + "% s" % repr(input_unit))

    # Add criterion with TimedArray
    output_traces = TimedArray(output.transpose(), dt=dt)

    error_unit = output.dim**2
    model = model + Equations('total_error : %s' % repr(error_unit))

    # Population size for differential evolution
    neurons = NeuronGroup(Ntraces * n_samples, model, method=method, name='neurons')
    neurons.namespace['input_var'] = input_traces
    neurons.namespace['output_var'] = output_traces
    neurons.namespace['t_start'] = t_start
    neurons.namespace['Nsteps'] = Nsteps
    neurons.namespace['Ntraces'] = Ntraces

    # initalize the values
    if param_init:
        neurons.set_states(param_init)

    # Record error
    neurons.run_regularly('total_error +=  (' + output_var + '-output_var\
                          (t,i % Ntraces))**2 * int(t>=t_start)', when='end')


    # Initialize the values
    def get_param_dic(parameters):
        parameters = array(parameters)

        d = dict()

        for name, value in zip(parameter_names, parameters.T):
            d[name] = (ones((Ntraces, n_samples)) * value).T.flatten()
        return d

    def calc_error():
        err = neurons.total_error/int((duration-t_start)/defaultclock.dt)
        err = mean(err.reshape((n_samples, Ntraces)), axis=1)

        return array(err)

    # Set up Simulator and Optimizer
    optimizer.initialize(parameter_names, **params)
    monitor = StateMonitor(neurons, output_var, record=True, name='monitor')

    network = Network()
    network.add(neurons, monitor)
    simulator.initialize(network)

    # Run Optimization Loop
    for k in range(n_rounds):
        parameters = optimizer.ask(n_samples=n_samples)
        d_param = get_param_dic(parameters)
        simulator.run(duration, d_param, parameter_names)
        out = getattr(simulator.network['monitor'], output_var)
        # out2 = getattr(monitor, output_var)

        if isinstance(metric, Metric):
            errors = metric.calc(out, output, Ntraces)
        else:
            errors = calc_error()

        optimizer.tell(parameters, errors)
        res = optimizer.recommend()

        # create output variables
        result_dict = make_dic(parameter_names, res)
        index_param = where(array(parameters) == array(res))
        ii = index_param[0]
        # TODO: fix error
        # print('index_param', index_param)
        error = errors[ii]  # TODO: re-check

        if verbose:
            print('round {} with error {}'.format(k, error))

    return result_dict, error
