import os
from numpy import mean, ones, array, where, atleast_1d
from brian2 import NeuronGroup, run, defaultclock, second, set_device, device
from brian2.input import TimedArray
from brian2.equations.equations import Equations

__all__ = ['fit_traces_standalone']

def initialize_parameter(variableview, value):
    variable = variableview.variable
    array_name = device.get_array_name(variable)
    static_array_name = device.static_array(array_name, value)
    device.main_queue.append(('set_by_array', (array_name,
                                               static_array_name,
                                               False)))
    return static_array_name


def set_parameter_value(identifier, value):
    atleast_1d(value).tofile(os.path.join(device.project_dir,
                                          'static_arrays',
                                          identifier))

def run_again():
    device.run(device.project_dir, with_output=False, run_args=[])

def set_states(init_dict, values):
    # TODO: add a param checker
    for obj_name, obj_values in values.items():
        set_parameter_value(init_dict[obj_name], obj_values)


def fit_traces_standalone(model=None,
                        input_var=None,
                        input=None,
                        output_var=None,
                        output=None,
                        dt=None,
                        maxiter=None,
                        t_start=0*second,
                        method=('linear', 'exponential_euler', 'euler'),
                        optimizer=None,
                        method_opt='DE',
                        n_samples=10,
                        n_rounds=1,
                        kwds_opt={},
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
        Child of Optimizer class, specific for each library.
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
    Errors array for each set of parameters (RMS).
    '''
    set_device('cpp_standalone', directory='parallel', clean=False)


    parameter_names = model.parameter_names

    for param in params.keys():
        if (param not in model.parameter_names):
            raise Exception("Parameter %s must be defined as a parameter in \
                             the model" % param)
    for param in model.parameter_names:
        if (param not in params):
            raise Exception("Bounds must be set for parameter %s" % param)

    bounds = []
    for name in parameter_names:
        bounds.append(params[name])     # Check parameter name

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')
    defaultclock.dt = dt

    # Check input variable
    if input_var not in model.identifiers:
        raise Exception("%s is not an identifier in the model" % input_var)

    Nsteps, Ntraces = input.shape
    duration = Nsteps * dt
    # Check output variable
    if output_var not in model.names:
        raise Exception("%s is not a model variable" % output_var)
    if output.shape != input.shape:
        raise Exception("Input and output must have the same size")

    # Replace input variable by TimedArray
    input_traces = TimedArray(input, dt = dt)
    input_unit = input.dim
    model = model + Equations(input_var + '= input_var(t,i % Ntraces) :\
                              '+ "% s" % repr(input_unit))

    # Add criterion with TimedArray
    output_traces = TimedArray(output, dt=dt)
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

    # Initialize the values
    params_init = dict()
    for name in parameter_names:
        params_init[name] = initialize_parameter(neurons.__getattr__(name),
                                                 ones((Ntraces, n_samples)))

    run(duration, namespace={})

    def calc_error(parameters):
        parameters = array(parameters)
        d = dict()
        for name, value in zip(parameter_names, parameters.T):
            d[name] = (ones((Ntraces, n_samples)) * value[0]).T.flatten()

        set_states(params_init, d)
        run_again()

        err = neurons.total_error/int((duration-t_start)/defaultclock.dt)
        err = mean(err.reshape((n_samples, Ntraces)), axis=1)

        return array(err)

    # set up the optimizer and get recommendation
    optim = optimizer(method=method_opt, parameter_names=parameter_names,
                      bounds=bounds, **kwds_opt)


    for _ in range(n_rounds):
        parameters = optim.ask(n_samples=n_samples)
        errors = calc_error(parameters)
        optim.tell(parameters, errors)
        res = optim.recommend()

        # create output variables
        resdict = dict()
        for name, value in zip(parameter_names, res):
            resdict[name] = value

        index_param = where(array(parameters) == array(res))

        ii = index_param[0]
        error = errors[ii][0]

        # TODO: add feedback and tolerance

    return resdict, error
