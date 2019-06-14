from brian2 import *
from brian2tools import *

set_device('cpp_standalone', directory='parallel', clean=False)

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


n_opt = NevergradOptimizer()

# pass parameters to the NeuronGroup
res, error = fit_traces_standalone(model=model, input_var='v', output_var='I',
                                   input=input_traces, output=output_traces,
                                   dt=0.1*ms, optimizer=n_opt,
                                   n_rounds=2, n_samples=10,
                                   g=[1*nS, 30*nS], E=[-20*mV, 100*mV],)

print(res, error)
