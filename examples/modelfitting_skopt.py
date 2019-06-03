from brian2 import *
from brian2tools import *
from skopt import Optimizer
from skopt.space import Real
from sklearn.externals.joblib import Parallel, delayed


# create input and output
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

# setup the skopt optimizer
optimizer = Optimizer(
    dimensions=[Real(-5.0, 5.0), Real(0.0, 10.0)],
    random_state=1,
    base_estimator='gp'
)

parameters = optimizer.ask(n_points=10)


# pass parameters to the NeuronGroup
errors = fit_traces_ask_tell(model = model, input_var = 'v', output_var = 'I',\
                            input = input_traces, output = output_traces, dt = 0.1*ms,
                            g = [1*nS, 30*nS], E = [-20*mV,100*mV], update=parameters)


# give information to the optimizer
optimizer.tell(parameters, errors.tolist());

xi = optimizer.Xi
yii = np.array(optimizer.yi)

# print the parameters, errors and result
for d in zip(parameters, errors):
    print(d)

print(xi[yii.argmin()])
