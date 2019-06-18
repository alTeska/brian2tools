from numpy import mean, ones, array, where
from brian2 import NeuronGroup, store, restore, run, defaultclock, second, StateMonitor
from brian2.input import TimedArray
from brian2.equations.equations import Equations
from brian2 import *


__all__ = ['fit_traces_ask_tell']


def fit_traces_ask_tell(model=None,
                        input_var=None,
                        input=None,
                        output_var=None,
                        output=None,
                        dt=None,
                        maxiter=None,
                        t_start=0*second,
                        method=('linear', 'exponential_euler', 'euler'),
                        optimizer=None,
                        n_samples=10,
                        n_rounds=1,
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
    output_var : string
        Output variable name.
    input : input data as a 2D array
    output : output data as a 2D array
    dt : time step
    maxiter : int, optional
        Maximum number of iterations.
    method: string, optional
        Integration method
    t_start: starting time of error measurement.

    optimizer: ~brian2tools.modelfitting.Optimizer children
        Pre-initialized child of Optimizer class, specific for each library.
    method_opt: string
        Optimization method, to be chosen within each library.
    n_samples: int
        Number of parameter samples to be optimized over.
    n_rounds: int
        Number of rounds to optimize over. (feedback provided over each round)

    TODO:
        -tolerance

    Returns
    -------
    resdict : dict
        dictionary with best parameter set
    error: float
        error value for best parameter set
    '''

    parameter_names = model.parameter_names

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')
    defaultclock.dt = dt

    # Check input variable
    if input_var not in model.identifiers:
        raise Exception("%s is not an identifier in the model" % input_var)

    Nsteps, Ntraces = input.shape
    duration = Nsteps*dt
    # Check output variable
    if output_var not in model.names:
        raise Exception("%s is not a model variable" % output_var)
    if output.shape != input.shape:
        raise Exception("Input and output must have the same size")

    # Replace input variable by TimedArray
    input_traces = TimedArray(input, dt = dt)
    # input_traces = TimedArray(input.transpose(), dt = dt)

    input_unit = input.dim
    model = model + Equations(input_var + '= input_var(t,i % Ntraces) :\
                              '+ "% s" % repr(input_unit))

    # Add criterion with TimedArray
    output_traces = TimedArray(output, dt=dt)
    # output_traces = TimedArray(output.transpose(), dt=dt)

    error_unit = output.dim**2
    model = model + Equations('total_error : %s' % repr(error_unit))

    # Population size for differential evolution
    neurons = NeuronGroup(Ntraces * n_samples, model, method=method)
    neurons.namespace['input_var'] = input_traces
    neurons.namespace['output_var'] = output_traces
    neurons.namespace['t_start'] = t_start
    neurons.namespace['Ntraces'] = Ntraces

    # Record error
    neurons.run_regularly('total_error +=  (' + output_var + '-output_var(t,i % Ntraces))**2 * int(t>=t_start)',
                          when='end')


    # Store for reinitialization
    store()

    def calc_error(parameters):
        d = dict()
        parameters = array(parameters)

        for name, value in zip(parameter_names, parameters.T):
            d[name] = (ones((Ntraces, n_samples)) * value[0]).T.flatten()

        restore()
        neurons.set_states(d, units=False)

        monitor = StateMonitor(neurons, [output_var, input_var, 'total_error'], record=True)

        run(duration, namespace={})

        err = neurons.total_error/int((duration-t_start)/defaultclock.dt)
        err = mean(err.reshape((n_samples, Ntraces)), axis=1)

        return array(err), monitor

    # set up the optimizer and get recommendation
    optimizer.initialize(parameter_names, **params)

    for _ in range(n_rounds):
        parameters = optimizer.ask(n_samples=n_samples)

        errors, monitor = calc_error(parameters)
        tot_err = getattr(monitor, 'total_error' + '_')
        inp = getattr(monitor, input_var + '_')
        out = getattr(monitor, output_var + '_')

        # fig, ax = plt.subplots(nrows=3)
        # ax[0].plot(out.transpose())
        # ax[1].plot(inp.transpose())
        # ax[2].plot(tot_err.transpose())
        # plt.show()

        optimizer.tell(parameters, errors)
        res = optimizer.recommend()

        # create output variables
        resdict = dict()
        for name, value in zip(parameter_names, res):
            resdict[name] = value

        index_param = where(array(parameters) == array(res))

        ii = index_param[0]
        error = errors[ii][0]
        # TODO: add feedback and tolerance

    return resdict, error
