import numpy as np
from brian2 import *
from brian2tools import *

dt = 0.01 * ms
defaultclock.dt = dt

# Generate a step-current input and an "experimental" voltage trace
input_current1 = np.hstack([np.zeros(int(5*ms/dt)), np.ones(int(5*ms/dt))*5, np.zeros(int(5*ms/dt))])* 5 *nA
input_current0 = np.hstack([np.zeros(int(5*ms/dt)), np.ones(int(5*ms/dt))*10, np.zeros(int(5*ms/dt))]) * 5 * nA

input_current2 = np.stack((input_current0, input_current1))
I = TimedArray(input_current0, dt=dt)

# C = 1*nF
# gL = 30*nS
EL = -70*mV
VT = -50*mV
DeltaT = 2*mV
eqs = Equations('''
    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I(t))/C : volt
    gL: siemens (constant)
    C: farad (constant)
    ''')

group = NeuronGroup(1, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    method='exponential_euler')
group.v = -70 *mV
group.set_states({'gL': [30*nS], 'C':[1*nF]})

monitor0 = StateMonitor(group, 'v', record=True)
smonitor0  = SpikeMonitor(group)

run(60*ms)

voltage0 = monitor0.v[0]/mV
out_spikes0 = getattr(smonitor0, 't') / ms

start_scope()
I = TimedArray(input_current1, dt=dt)
group1 = NeuronGroup(1, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    method='exponential_euler')
group1.v = -70 *mV
group1.set_states({'gL': [30*nS], 'C':[1*nF]})

monitor1 = StateMonitor(group1, 'v', record=True)
smonitor1  = SpikeMonitor(group1)
run(60*ms)

out_spikes1 = getattr(smonitor1, 't') / ms
voltage1 = monitor1.v[0]/mV


inp_trace0 = np.array([input_current0])
inp_trace1 = np.array([input_current1])

inp_trace = np.concatenate((inp_trace0, inp_trace1))
out_spikes = np.array(([out_spikes0, out_spikes1]))


# FIT SPIKES
start_scope()
eqs_fit = Equations('''
    dv/dt = (gL*(EL-v)+gL*DeltaT*exp((v-VT)/DeltaT) + I)/C : volt
    gL: siemens (constant)
    C: farad (constant)
    ''',
    EL = -70*mV,
    VT = -50*mV,
    DeltaT = 2*mV,
    # C=1*nF
    )

n_opt = NevergradOptimizer('PSO')
metric = GammaFactor(100*ms, dt)


# pass parameters to the NeuronGroup
result_dict, error = fit_spikes(model=eqs_fit, input_var='I', dt=dt,
                                input=inp_trace * amp, output=out_spikes,
                                n_rounds=1, n_samples=10, optimizer=n_opt,
                                metric=metric,
                                threshold='v > -50*mV',
                                reset='v = -70*mV',
                                method='exponential_euler',
                                param_init={'v': -70*mV},
                                gL=[20*nS, 40*nS],
                                C = [0.5*nF, 1.5*nF]
                                )

# print(result_dict, error)
print('goal:', {'gL': [30*nS], 'C':[1*nF]})
print('results:', result_dict['C']*farad, result_dict['gL']*siemens)

res = {'gL': [result_dict['gL']*siemens], 'C': [result_dict['C']*farad]}

# visualization of the results
start_scope()
group_1 = NeuronGroup(1, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    method='exponential_euler')
group_1.v = -70 * mV

group_1.set_states(res)

monitor_1 = StateMonitor(group_1, 'v', record=True)
smonitor_1  = SpikeMonitor(group_1)

run(60*ms)
voltage_1 = monitor_1.v[0]/mV

start_scope()
I = TimedArray(input_current0, dt=dt)
group_0 = NeuronGroup(1, eqs,
                    threshold='v > -50*mV',
                    reset='v = -70*mV',
                    method='exponential_euler')
group_0.v = -70 * mV

group_0.set_states(res)

monitor_0 = StateMonitor(group_0, 'v', record=True)
smonitor_0  = SpikeMonitor(group_0)

run(60*ms)
voltage_0 = monitor_0.v[0]/mV


fig, ax = plt.subplots(nrows=2)
ax[0].plot(voltage0);
ax[0].plot(voltage_0);
ax[1].plot(voltage1);
ax[1].plot(voltage_1);
plt.show()
