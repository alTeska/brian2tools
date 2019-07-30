Model fitting
=============

The `brian2tools` offers model fitting package, that allows for data driven optimization of custom
models. We offer the users a toolbox, that allows the user to find the best fit of the parameters
for recorded traces and spike trains.

Model provides two 
.. contents::
    Overview
    :local:


## How it works
Model fitting requires two components:
 - A **metric**: to compare results and decide which one is the best
 - An **optimization** algorithm: to decide which parameter combinations to try

Each scripts require
```
opt = Optimizer()
metric = Metric()

res, error = fit_traces(metric=metric, optimizer=opt, ...)
res, error = fit_spikes(metric=metric, optimizer=opt, ...)
```

The proposed solution is developed using a modular approach, where both the optimization method and
metric to be optimized can be easily swapped out.


### Optimizer

- gradient free methods - global methods
    * evolutionary algorithms
    * genetic algorithms
    * bayesian optimization
    * ...

By default, we support a range of global *derivative-free optimization* methods (provided by the library **Nevergrad**)
as well as *Bayesian Optimization* for black box functions (provided by **Scikit-Optimize**).

#### Follows `ask()/tell()` interface

Abstract `class Optimizer` prepared for different back-end libraries!

User can plug in different optimization tool, as long as it follows``` ask() / tell``` interface:

```
parameters = optimizer.ask()

errors = simulator.run(parameters)

optimizer.tell(parameters, errors)
results = optimizer.recommend()
```

### Provided libraries and methods:
#### Nevergrad
offers an extensive collection of algorithms that do not require gradient computation

<img src='./images/animation_gray_border.gif'  width="900" height="900" align="right"/>

**Method examples:**
- Differential evolution.
- Sequential quadratic programming.
- FastGA.
- Covariance matrix adaptation.
- Particle swarm optimization.
- ...

#### Scikit-Optimize (skopt)
<img src='./images/skopt_example_1to3.png'  width="450" height="450" align="right"/>

- bayesian-optimization
- library to minimize (very) expensive and noisy black-box functions
- implements several methods for sequential model-based
  optimization
- based no scikit-learn minimize function

https://github.com/scikit-optimize

PyData talk:
https://www.youtube.com/watch?v=DGJTEBt0d-s

#### https://github.com/facebookresearch/nevergrad
### Metric

- Mean Square Error
For metrics, user can select one of the available metrics, eg.: GammaFactor, or easily plug in a code
extension with a custom metric.

- additionally an offline MSE can be calculated

### Standalone mode

Both Brian and the Model Fitting Toolbox are designed to be easily used and save time through automatic
parallelization of the simulations using code generation.


### Callback function

The feedback provided by the fitting function is designed with the same principle in mind and can also
be easily extended to fulfil the individual requirements.

## Local Gradient Optimization
gradient based methods - local application

Coming soon...


### Utils: generate fits
```
fits = generate_fits(model=model, params=res, input=input_current * amp,
                     input_var='I', output_var='v', param_init={'v': -30*mV},
                     dt=dt)
```

## Examples






## API




Create a Brian Model Fitting toolbox that works with traces and spike trains
- vectorization and model flexibility from brian
- modularity:
    * multiple optimization methods (libraries)
    * multiple metrics
    * custom callback function available
- find a good balance between flexible system (user can define whatever they want) and convenience (provide a few standard metrics)

## Requires:
- an **optimization algorithm**

- a **metric**

### makes use of Brian parallelisation and flexibility
