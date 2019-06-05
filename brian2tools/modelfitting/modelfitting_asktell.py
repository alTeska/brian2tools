from numpy import mean, ones, array, shape
from brian2.equations.equations import (DIFFERENTIAL_EQUATION, Equations,
                                        SingleEquation, PARAMETER)
from brian2.input import TimedArray
from brian2 import NeuronGroup, StateMonitor, store, restore, run, defaultclock, second, Quantity
from brian2.stateupdaters.base import StateUpdateMethod


__all__=['fit_traces_ask_tell']


def fit_traces_ask_tell(model=None,
               input_var=None,
               input = None,
               output_var = None,
               output = None,
               dt = None,
               maxiter = None,
               t_start = 0*second,
               method = ('linear', 'exponential_euler', 'euler'),
               update=None,
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
    update: list of parameters to evaluate over
    Returns
    -------
    Errors array for each set of parameters (RMS).
    '''


    parameter_names = model.parameter_names

    for param in params.keys():
        if (param not in model.parameter_names):
            raise Exception("Parameter %s must be defined as a parameter in the model" % param)
    for param in model.parameter_names:
        if (param not in params):
            raise Exception("Bounds must be set for parameter %s" % param)

    bounds = []
    for name in parameter_names:
        bounds.append(params[name])# Check parameter names
    print('bounds', bounds)

    # dt must be set
    if dt is None:
        raise Exception('dt (sampling frequency of the input) must be set')
    defaultclock.dt = dt


    # Check input variable
    if input_var not in model.identifiers:
        raise Exception("%s is not an identifier in the model" % input_var)

    Nsteps, Ntraces = input.shape
    # Ntraces, Nsteps = input.shape
    duration = Nsteps*dt
    # Check output variable
    if output_var not in model.names:
        raise Exception("%s is not a model variable" % output_var)
    if output.shape!=input.shape:
        raise Exception("Input and output must have the same size")

    # This only works because the equations are completely self-contained
    # TODO: This will not work like this for models with refractoriness
    state_update_code = StateUpdateMethod.apply_stateupdater(model, {},
                                                             method=method)
    # Remove all differential equations from the model (they will be updated
    # explicitly)
    model_without_diffeq = Equations([eq for eq in model.ordered
                                      if eq.type != DIFFERENTIAL_EQUATION])
    # Add a parameter for each differential equation
    diffeq_params = Equations([SingleEquation(PARAMETER, varname, model.dimensions[varname])
                               for varname in model.diff_eq_names])

    # Our new model:
    model = model_without_diffeq + diffeq_params

    # Replace input variable by TimedArray
    input_traces = TimedArray(input, dt = dt)
    input_unit = input.dim
    model = model + Equations(input_var + '= input_var(t,i % Ntraces) : '+ "% s" % repr(input_unit))

    # Add criterion with TimedArray
    output_traces = TimedArray(output, dt = dt)
    error_unit = output.dim**2
    model = model + Equations('total_error : %s' % repr(error_unit))

    # Population size for differential evolution
    # (actually in scipy's algorithm this is popsize * nb params)

    popsize, _ = shape(update)
    # N = popsize * len(parameter_names)

    neurons = NeuronGroup(Ntraces * popsize, model, method = method)
    neurons.namespace['input_var'] = input_traces
    neurons.namespace['output_var'] = output_traces
    neurons.namespace['t_start'] = t_start
    neurons.namespace['Ntraces'] = Ntraces

    # Record error
    neurons.run_regularly('total_error +=  (' + output_var + '-output_var(t,i % Ntraces))**2 * int(t>=t_start)',
                          when='end')

    # Add the code doing the numerical integration
    neurons.run_regularly(state_update_code, when='groups')

    # Store for reinitialization
    store()

    def calc_error(update):

        d = dict()
        update = array(update)

        for name, value in zip(parameter_names, update.T):
            d[name] = (ones((Ntraces, popsize)) * update.T[0]).T.flatten()

        restore()
        neurons.set_states(d, units=False)
        run(duration, namespace = {})

        e = neurons.total_error/int((duration-t_start)/defaultclock.dt)
        e = mean(e.reshape((popsize, Ntraces)),axis=1)

        return array(e)

    errors = calc_error(update)

    return errors
