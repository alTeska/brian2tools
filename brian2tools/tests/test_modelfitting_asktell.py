from brian2 import *
from brian2tools import *

from nevergrad.optimization import optimizerlib
from concurrent import futures
from nevergrad import instrumentation as inst

candidates, parameters = [], []

# setup input and output
input_traces = zeros((10,1))*volt
for i in range(1):
    input_traces[1:,i]=i*10*mV

# Create target current traces
output_traces = 10*nS*input_traces


model = Equations('''
    I = g*(v-E) : amp
    g : siemens (constant)
    E : volt (constant)
    ''')

# setup the nevergrad optimizer
arg1 = inst.var.Array(1).bounded(-5, 5).asscalar()
arg2 = inst.var.Array(1).bounded(0, 10).asscalar()
instrum = inst.Instrumentation(arg1, arg2)
optim = optimizerlib.registry['DE'](instrumentation=instrum, budget=10000)

for _ in range(10):
    cand = optim.ask()
    candidates.append(cand)
    parameters.append(list(cand.args))



# pass parameters to the NeuronGroup
errors = fit_traces_ask_tell(model = model, input_var = 'v', output_var = 'I',\
                            input = input_traces, output = output_traces, dt = 0.1*ms,
                            g = [1*nS, 30*nS], E = [-20*mV,100*mV], update=parameters)



for i, candidate in enumerate(candidates):
    value = errors[i]
    optim.tell(candidate, value)

    print(candidate, value)

ans = optim.provide_recommendation()
list(ans.args)
