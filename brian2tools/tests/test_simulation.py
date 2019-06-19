'''
Test the simulation class
'''
from brian2tools import Simulation, RuntimeSimulation, CPPStandaloneSimulation


def test_init():
    Simulation()
    RuntimeSimulation()
    CPPStandaloneSimulation()
