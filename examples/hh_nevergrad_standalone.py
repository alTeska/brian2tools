from brian2 import *
from brian2tools import *


prefs.codegen.target = 'cython'  # weave is not multiprocess-safe!

# Parameters
area = 20000*umetre**2
Cm = 1*ufarad*cm**-2 * area
El = -65*mV
EK = -90*mV
ENa = 50*mV
VT = -63*mV

dt = 0.01*ms

# Generate a step-current input and an "experimental" voltage trace
input_current = np.hstack([np.zeros(int(5*ms/dt)), np.ones(int(5*ms/dt)), np.zeros(int(5*ms/dt))])*nA

params_correct = {'gl': float(5e-5*siemens*cm**-2 * area),
                  'g_na': float(100*msiemens*cm**-2 * area),
                  'g_kd': float(30*msiemens*cm**-2 * area)}

defaultclock.dt = dt

# The model
eqsHH = Equations('''
dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I(t))/Cm : volt
dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
    (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
    (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
    (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
g_na : siemens (constant)
g_kd : siemens (constant)
gl   : siemens (constant)
''')

I = TimedArray(input_current, dt=dt)

G = NeuronGroup(1, eqsHH, method='exponential_euler')
G.v = El
G.set_states(params_correct, units=False)
mon = StateMonitor(G, 'v', record=0)
run(25*ms)

voltage = mon.v[0]/mV
voltage += np.random.randn(len(voltage))


inp_trace = np.array([input_current])
n0, n1 = inp_trace.shape

out_trace = np.array(voltage[:n1])


# Model for modelfitting
eqs = Equations(
'''
dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
    (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
    (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
    (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
g_na : siemens (constant)
g_kd : siemens (constant)
gl   : siemens (constant)
''',
Cm=1*ufarad*cm**-2 * area, El=-65*mV, EK=-90*mV, ENa=50*mV, VT=-63*mV)

set_device('cpp_standalone', directory='parallel', clean=False)

n_opt = NevergradOptimizer()
metric = MSEmetric()

# pass parameters to the NeuronGroup
res, error = fit_traces_standalone(model=eqs, input_var='I', output_var='v',
                                   input=inp_trace * amp, output=[out_trace]*mV, dt=dt,
                                   method='exponential_euler',
                                   gl=[1e-8*siemens*cm**-2 * area, 1e-4*siemens*cm**-2 * area],
                                   g_na=[1*msiemens*cm**-2 * area, 200*msiemens*cm**-2 * area],
                                   g_kd=[1*msiemens*cm**-2 * area, 100*msiemens*cm**-2 * area],
                                   n_rounds=2, n_samples=5, optimizer=n_opt, metric=metric)


# give information to the optimizer
print('correct:', params_correct, '\n output:', res)
print('error', error)

start_scope()
fits = generate_fits(model=eqs, method='exponential_euler', params=res,
                     input=inp_trace * amp, input_var='I', output_var='v', dt=dt)



# visualization of the results
# start_scope()
# G = NeuronGroup(1, eqsHH, method='exponential_euler')
# G.v = El
# G.set_states(res, units=False)
# mon = StateMonitor(G, 'v', record=0)
# # mon = StateMonitor(G, ['v', 'I'], record=0)
# run(25*ms)
#
# voltage1 = mon.v[0]/mV
#
#
# plt.figure()
# plot(np.arange(len(voltage))*dt/ms, voltage);
# plot(np.arange(len(voltage1))*dt/ms, voltage1);
# plt.title('nevergrad optimization')
# plt.savefig('plots/hh_nevergrad.png')
# plt.show()
