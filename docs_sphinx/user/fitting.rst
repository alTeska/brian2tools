Model fitting
=============

The `brian2tools` offers model fitting package, that allows for data driven optimization of custom
models. We offer the users a toolbox, that allows the user to find the best fit of the parameters
for recorded traces and spike trains. Just like Brian he Model Fitting Toolbox is designed to be
easily used and save time through automatic parallelization of the simulations using code generation.

Model provides two functions:
 - `fit_spikes()`
 - `fit_traces()`


That accept the model, data as an input and returns  best fit of parameters and corresponding error.


.. contents::
    Overview
    :local:


In following documentation we assume that ``brian2tools`` has been imported like this:

.. code:: python

    from brian2tools import *


How it works
------------

Model fitting requires two components:
 - A **metric**: to compare results and decide which one is the best
 - An **optimization** algorithm: to decide which parameter combinations to try

Each scripts requires following initialization:

.. code:: python

  opt = Optimizer()
  metric = Metric()

  params, error = fit_traces(metric=metric, optimizer=opt, ...)
  params, error = fit_spikes(metric=metric, optimizer=opt, ...)


The proposed solution is developed using a modular approach, where both the optimization method and
metric to be optimized can be easily swapped out by the users custom implementation.

Both fitting functions require model defined as ``Equation`` object, that has parameters that will be
optimized specified as constants in a following way:

.. code:: python

  '''
  ...
  g_na : siemens (constant)
  g_kd : siemens (constant)
  gl   : siemens (constant)
  '''

In case of spiking neurons,

Example of `fit_traces()` with all of the necessary arguments:

.. code:: python

  params, error = fit_traces(model=model,
                             input_var='I',
                             output_var='v',
                             input=inp_trace,
                             output=out_trace,
                             dt=0.1*ms,
                             optimizer=n_opt,
                             metric=metric,
                             n_rounds=1, n_samples=5,
                             gl=[1e-8*siemens*cm**-2 * area, 1e-3*siemens*cm**-2 * area],)



Optimizer
---------

Optimizer classes uses gradient free global optimization methods
(evolutionary algorithms, genetic algorithms, Bayesian optimization)


Follows `ask()/tell()` interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User can plug in different optimization tool, as long as it follows ```ask() / tell``` interface.
Abstract `class Optimizer` prepared for different back-end libraries. All of the optimizer specific
arguments have to be provided upon optimizers initialization.


```ask() / tell``` interface:

.. code:: python

  parameters = optimizer.ask()

  errors = simulator.run(parameters)

  optimizer.tell(parameters, errors)
  results = optimizer.recommend()


Provided libraries and methods:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Nevergrad**
Offers an extensive collection of algorithms that do not require gradient computation.
Nevergrad optimizer can be specified in the following way:

.. code:: python

  opt = NevergradOptimizer(method='PSO')

where method input is a string with specific optimization algorithm.

**Available methods include:**
 - Differential evolution. ['DE']
 - Covariance matrix adaptation.['CMA']
 - Particle swarm optimization.['PSO']
 - Sequential quadratic programming.['SQP']


Nevergrad is not yet documented, to check availible methods use following code:

.. code:: python

  from nevergrad.optimization import registry
  print(sorted(registry.keys()))

Source code:

https://github.com/facebookresearch/nevergrad

Important notes:
 - TODO: number of samples per round in Nevergrad optimization methods is limited to 30,
   to increase it has to be manually changed


**2. Scikit-Optimize (skopt)**
Skopt implements several methods for sequential model-based ("blackbox") optimization
and focuses on bayesian methods. Algorithms are based on scikit-learn minimize function.

**Available Methods:**
 - Gaussian process-based minimization algorithms ['GP']
 - Sequential optimization using gradient boosted trees ['GBRT']
 - Sequential optimisation using decision trees ['ET']
 - Random forest regressor ['RF']

User can also provide a custom made sklearn regressor!

.. code:: python

  opt = SkoptOptimizer(method='GP')

Documentation:
https://github.com/scikit-optimize

PyData talk:
https://www.youtube.com/watch?v=DGJTEBt0d-s



Metric
------
For metrics, user can select one of the available metrics, eg.: GammaFactor, or easily plug in a code
extension with a custom metric.



**1. Mean Square Error**

.. math:: MSE ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2} $$

.. code:: python

  metric = MSEMetric()

also calculated offline with ``metric=None`` as input


**2. GammaFactor - for fit_spikes.**

.. math:: \Gamma = \left (\frac{2}{1-2\delta r_{exp}}\right) \left(\frac{N_{coinc} - 2\delta N_{exp}r_{exp}}{N_{exp} + N_{model}}\right)$$

:math:`N_{coinc}$` - number of coincidences

:math:`N_{exp}` and :math:`N_{model}`- number of spikes in experimental and model spike trains

:math:`r_{exp}` - average firting rate in experimental train

:math:`2 \delta N_{exp}r_{exp}` - expected number of coincidences with a poission process

.. code:: python

  metric = GammaFactor(delta=10*ms, dt=0.1*ms)


Features
--------
Standalone mode
~~~~~~~~~~~~~~~

 To run the

.. code:: python

  set_device('cpp_standalone', directory='parallel', clean=False)



Callback function
~~~~~~~~~~~~~~~~~

The feedback provided by the fitting function is designed with the same principle in mind and can also
be easily extended to fulfil the individual requirements.

boolean or function

``callback = True`` - returns default print out

.. code:: python

  def callback(res, errors, parameters, index):
      print('index {} errors minimum: {}'.format(index, min(errors)) )

Additional inputs
~~~~~~~~~~~~~~~~~


Local Gradient Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Additional local optimization with use of gradient methods can be applied.
Coming soon...


Utils: generate fits
--------------------

In toolboxes utils we provided a helper function that will generate required traces
based on same model and input. To be used after fitting.

.. code:: python

  fits = generate_fits(model=model, params=res, input=input_current,
                       input_var='I', output_var='v', param_init={'v': -30*mV},
                       dt=0.1*ms)


Simple Examples
---------------


fit_spikes
~~~~~~~~~~

.. code:: python

  n_opt = NevergradOptimizer('DE')
  metric = GammaFactor(dt, 60*ms)


  params, error = fit_spikes(model=eqs, input_var='I', dt=0.1*ms,
                             input=inp_traces, output=out_spikes,
                             n_rounds=2, n_samples=30, optimizer=n_opt,
                             metric=metric,
                             threshold='v > -50*mV',
                             reset='v = -70*mV',
                             method='exponential_euler',
                             param_init={'v': -70*mV},
                             gL=[20*nS, 40*nS],
                             C = [0.5*nF, 1.5*nF])



fit_traces
~~~~~~~~~~

.. code:: python

  n_opt = NevergradOptimizer(method='PSO')
  metric = MSEMetric()

  params, error = fit_traces(model=model,
                             input_var='I',
                             output_var='v',
                             input=inp_trace,
                             output=out_trace,
                             param_init={'v': -65*mV},
                             method='exponential_euler',
                             dt=0.1*ms,
                             optimizer=n_opt,
                             metric=metric,
                             callback=True,
                             n_rounds=1, n_samples=5,
                             gl=[1e-8*siemens*cm**-2 * area, 1e-3*siemens*cm**-2 * area],
                             g_na=[1*msiemens*cm**-2 * area, 2000*msiemens*cm**-2 * area],
                             g_kd=[1*msiemens*cm**-2 * area, 1000*msiemens*cm**-2 * area],)
