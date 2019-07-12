
def fit_spikes(model=None,
               input_var=None,
               input=None,
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
               reset=None, refractory=False, threshold=None,
               **params):
    """Fit spikes"""

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
                raise Exception("%s is not a model variable or an identifier \
                                in the model")

    # Check input variable
    if input_var not in model.identifiers:
        raise Exception("%s is not an identifier in the model" % input_var)

    if not isinstance(metric, Metric):
        raise Exception("metric has to be a child of class Metric")

    # Check output variable
    Ntraces, Nsteps = input.shape
    duration = Nsteps * dt

    # Replace input variable by TimedArray
    input_traces = TimedArray(input.transpose(), dt=dt)

    input_unit = input.dim
    model = model + Equations(input_var + '= input_var(t, i % Ntraces) :\
                               ' + "% s" % repr(input_unit))

    # Population size for differential evolution
    neurons = NeuronGroup(Ntraces * n_samples, model, method=method,
                          threshold=threshold, reset=reset,
                          refractory=refractory, name='neurons')

    neurons.namespace['input_var'] = input_traces
    neurons.namespace['t_start'] = t_start
    neurons.namespace['Nsteps'] = Nsteps
    neurons.namespace['Ntraces'] = Ntraces

    # initalize the values
    if param_init:
        neurons.set_states(param_init)

    # Set up Simulator and Optimizer
    monitor = SpikeMonitor(neurons, record=True, name='monitor')
    network = Network(neurons, monitor)

    optimizer.initialize(parameter_names, **params)
    simulator.initialize(network)

    # Run Optimization Loop
    for k in range(n_rounds):
        parameters = optimizer.ask(n_samples=n_samples)

        d_param = get_param_dic(parameters, parameter_names, Ntraces,
                                n_samples)

        simulator.run(duration, d_param, parameter_names)

        spike_trains = monitor.spike_trains()
        traces = []
        for i in arange(len(spike_trains)):
            trace = spike_trains[i] / ms
            traces.append(trace)

        errors = metric.calc(traces, output, Ntraces)
        optimizer.tell(parameters, errors)
        res = optimizer.recommend()

        # create output variables
        result_dict = make_dic(parameter_names, res)
        # create output variables
        result_dict = make_dic(parameter_names, res)
        error = min(errors)

        if verbose:
            print('round {} with error {}'.format(k, error))
            print("resulting parameters:", result_dict)

    return result_dict, error
