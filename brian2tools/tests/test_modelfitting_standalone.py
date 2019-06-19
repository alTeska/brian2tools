'''
Test the modelfitting module
'''
from brian2 import zeros, Equations
from brian2 import nS, mV, volt, ms
from brian2tools import fit_traces_standalone, NevergradOptimizer, SkoptOptimizer


input_traces = zeros((10, 5))*volt
for i in range(5):
    input_traces[5:, i] = i*10*mV

# Create target current traces
output_traces = 10*nS*input_traces
model = Equations('''
    I = g*(v-E) : amp
    g : siemens (constant)
    E : volt (constant)
    ''')


def test_import():
    fit_traces_standalone


def test_fit_traces():
    n_opt = NevergradOptimizer('DE')
    s_opt = SkoptOptimizer('gp')
    res, error = fit_traces_standalone(model=model, input_var='v', output_var='I',
                                       input=input_traces, output=output_traces,
                                       dt=0.1*ms, optimizer=n_opt,
                                       g=[1*nS, 30*nS], E=[-20*mV, 100*mV],)

    res, error = fit_traces_standalone(model=model, input_var='v', output_var='I',
                                       input=input_traces, output=output_traces,
                                       dt=0.1*ms, optimizer=s_opt,
                                       g=[1*nS, 30*nS], E=[-20*mV, 100*mV],)


# TODO: test different methods, test **kwds
