from brian2 import *
from brian2tools import *


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

n_opt = NevergradOptimizer(method='DE', popsize=10, budget=300)

# pass parameters to the NeuronGroup
res, error = fit_traces_ask_tell(model=model, input_var='v', output_var='I',
                                 input=input_traces, output=output_traces, dt=0.1*ms,
                                 g=[1*nS, 30*nS], E=[-20*mV,100*mV],
                                 optimizer=n_opt)

print(res, error)
