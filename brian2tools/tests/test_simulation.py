'''
Test the simulation class
'''
import numpy as np
from numpy.testing.utils import assert_equal, assert_raises
from brian2 import (Equations, NeuronGroup, StateMonitor, Network, ms,
                    defaultclock, device, get_device, start_scope)
from brian2tools import Simulation, RuntimeSimulation, CPPStandaloneSimulation
from brian2.devices.device import reset_device

model = Equations('''
    I = g*(v-E) : amp
    v = 10*mvolt :volt
    g : siemens (constant)
    E : volt (constant)
    ''')

dt = 0.1 * ms
defaultclock.dt = dt

neurons = NeuronGroup(1, model, name='neurons')
monitor = StateMonitor(neurons, 'I', record=True, name='monitor')

net = Network(neurons, monitor)


def test_init():
    Simulation()
    RuntimeSimulation()
    CPPStandaloneSimulation()


def test_initialize_parameter():
    pass


def test_initialize_neurons():
    pass


def test_run_again():
    pass


def test_set_parameter_value():
    pass


def test_set_states():
    pass


def test_initialize_simulation_standalone():
    start_scope()
    sas = CPPStandaloneSimulation()
    sas.initialize(net)
    assert(isinstance(sas.network, Network))


def test_run_simulation_standalone():
    dev = get_device()
    reset_device(dev)
    start_scope()

    neurons = NeuronGroup(1, model, name='neurons')
    monitor = StateMonitor(neurons, 'I', record=True, name='monitor')
    net = Network(neurons, monitor)

    device.has_been_run = False
    duration = 10*ms
    sas = CPPStandaloneSimulation()
    sas.initialize(net)

    sas.run(duration, {'g': 100, 'E': 10}, ['g', 'E'])
    I = getattr(sas.network['monitor'], 'I')
    assert_equal(np.shape(I), (1, duration/dt))


def test_initialize_simulation_runtime():
    start_scope()
    rts = RuntimeSimulation()
    rts.initialize(net)
    assert(isinstance(rts.network, Network))


def test_run_simulation_runtime():
    dev = get_device()
    reset_device(dev)
    start_scope()

    neurons = NeuronGroup(1, model, name='neurons')
    monitor = StateMonitor(neurons, 'I', record=True, name='monitor')
    net = Network(neurons, monitor)

    device.has_been_run = False
    duration = 10*ms
    rts = RuntimeSimulation()
    rts.initialize(net)

    rts.run(duration, {'g': 100, 'E': 10}, ['g', 'E'])
    I = getattr(rts.network['monitor'], 'I')
    assert_equal(np.shape(I), (1, duration/dt))
