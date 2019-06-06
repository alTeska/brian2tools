from brian2 import *
from brian2tools import *

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

# setup the nevergrad optimizer
n_opt = NevergradOptimizer(method='DE', parameter_names={'E', 'g'},
                           bounds=[[-5, 5], [0, 10]])

parameters = n_opt.ask(10)

# pass parameters to the NeuronGroup
errors = fit_traces_ask_tell(model = model, input_var = 'v', output_var = 'I',\
                            input = input_traces, output = output_traces, dt = 0.1*ms,
                            g = [1*nS, 30*nS], E = [-20*mV,100*mV], update=parameters)



# give information to the optimizer
n_opt.tell(parameters, errors)

ans = n_opt.recommend()

# show answers
for n in zip(parameters, errors):
    print(n)

print(list(ans.args))
