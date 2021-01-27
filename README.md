<h2 align="center">Bayesian modelling of FIDE world cup chess games</h2>

<div align="center">
  <!--Python version -->
  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/pypi/pyversions/fastai.svg"
      alt="Python version" />
  </a>
  <!--Project status -->
  <a href="https://github.com/maw501/bayesian-chess-prediction">
    <img src="https://img.shields.io/badge/Status-Under%20development-green.svg"
      alt="Status version" />
  </a>
  <!--Commits  -->
  <a href="https://github.com/maw501/bayesian-chess-prediction/commits/main">
    <img src="https://img.shields.io/github/last-commit/maw501/bayesian-chess-prediction.svg"
      alt="Status version" />
  </a>
</div>
<br />

## Overview

A repository using a Bayesian hierarchical model to try to predict the outcomes of FIDE chess world cup games in 2019. The model fits an ordered logistic regression model and learns a per player ability rating.

**Example of assessing impact of prior values on game outcomes:**

![Image](resources/prior_sim.png)

## Getting started

Aside from standard data science packages the main dependencies are [PyStan](https://pystan.readthedocs.io/en/latest/) and [ArviZ](https://arviz-devs.github.io/arviz/).

## Notebooks

There are example notebooks outlining the problem and parts of the Bayesian workflow. 

* [`0.overview_of_problem.ipynb`](https://nbviewer.jupyter.org/github/maw501/bayesian-chess-prediction/blob/main/notebooks/0.overview_of_problem.ipynb): introduces the dataset and a simple GLM model.
* [`1.fake_data_and_prior_simulation.ipynb`](notebooks/1.fake_data_and_prior_simulation.ipynb): simulating fake data and fitting a model to it and prior-predictive simulation.
* [`2.fit_ordered_logistic_model.ipynb`](notebooks/2.fit_ordered_logistic_model.ipynb) fit a model to the dataset.
* [`3.machine_learning_baseline.ipynb`](notebooks/3.machine_learning_baseline.ipynb): step back and fit another machine learning model to provide a benchmark for performance.

Less finished notebooks are in the `notebooks/investigations` folder. These include fitting a simpler model that doesn't learn a per player ability rating.

