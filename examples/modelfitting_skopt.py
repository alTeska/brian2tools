from brian2 import *
from brian2tools import *


# create input and output
input_traces = zeros((10,1))*volt
for i in range(1):
    input_traces[1:,i]=i*10*mV

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
                                 optimizer=SkoptOptimizer, method_opt='gp')

print(res, error)
