'''
Test the modelfitting module
'''
from brian2 import zeros, Equations
from brian2 import nS, mV, volt, ms
from brian2tools import fit_traces_ask_tell, NevergradOptimizer, SkoptOptimizer


def test_import():
    fit_traces_ask_tell

def test_fit_traces():
    # Create voltage traces for an activation experiment
    input_traces = zeros((10,5))*volt
    for i in range(5):
        input_traces[5:,i]=i*10*mV

    # Create target current traces
    output_traces = 10*nS*input_traces
    model = Equations('''
    I = g*(v-E) : amp
    g : siemens (constant)
    E : volt (constant)
    ''')
    res, error = fit_traces_ask_tell(model=model, input_var='v', output_var='I',
                                    input=input_traces, output=output_traces, dt=0.1*ms,
                                    g=[1*nS, 30*nS], E=[-20*mV,100*mV],
                                    optimizer=NevergradOptimizer, method_opt='DE')

    res, error = fit_traces_ask_tell(model=model, input_var='v', output_var='I',
                                    input=input_traces, output=output_traces, dt=0.1*ms,
                                    g=[1*nS, 30*nS], E=[-20*mV,100*mV],
                                    optimizer=SkoptOptimizer, method_opt='gp')

# TODO: test different methods, test **kwds