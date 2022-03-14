# sde-importance-sampling

This Python repository contains the implementation of importance sampling strategies for metastable diffusion processes. We build a framework to estimate path functionals of stochastic process following an overdamped Langevin equation.

## Contains

- Monte Carlo sampling
- Adapted metadynamics algorithm for path dependent problems
- Finite differences solver of the associated HJB equation
- Importance sampling given a control (null, random, meta, hjb)
- Related stochastic optimization control (SOC) problem with different control representations (ansatz type and neural networks) and different initializations (null, random, meta, hjb)

## Install

1) clone the repo 
```
git clone git@github.com:riberaborrell/sde_importance_sampling.git
```

2) set python version
```
pyenv local 3.9.7
```

3) create virtual environment and install required packages
```
make venv
```

4) activate venv
```
source venv/bin/activate
```

## Developement

in step 3) also install developement packages
```
make develop
```
