from brian2 import *
from brian2tools import *
from nevergrad.optimization import optimizerlib
from nevergrad import instrumentation as inst


candidates, parameters = [], []

# create input and output
input_traces = zeros((10,5))*volt
for i in range(5):
    input_traces[5:,i]=i*10*mV

output_traces = 10*nS*input_traces

model = Equations('''
    I = g*(v-E) : amp
    g : siemens (constant)
    E : volt (constant)
    ''')

# pass parameters to the NeuronGroup
res, error = fit_traces_ask_tell(model=model, input_var='v', output_var='I',
                                 input=input_traces, output=output_traces, dt=0.1*ms,
                                 g=[1*nS, 30*nS], E=[-20*mV,100*mV],
                                 optimizer=NevergradOptimizer, method_opt='DE')

print(res, error)
