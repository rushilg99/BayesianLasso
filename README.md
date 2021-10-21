This repository contains my work on the Bayesian Lasso, which evolved into an MSc thesis under the guidance of my supervisor Dr. Konstantinos Perrakis.
I am pleased to report that I was awarded 88%!
It was not an easy task to conduct my thesis in the shadow of the COVID-19 pandemic, but it was thoroughly enjoyable reading a variety of papers and learning a great deal
across the realms of Bayesian statistics, MCMC and Lasso regression. 

The project contained a nice balance between theory and applications; I've always been slightly pragmatic by nature and whilst I love learning about theory behind various
concepts, I also want to apply what I've learnt and make things. Hence, a significant portion of the project was devoted to building
Bayesian Lasso statistical models. Through a variety of simulations, we study the behaviour of point estimates of regression coefficients from the posterior distributions
(notably the posterior median). I also propose a class of novel Gamma hyperpriors (titled "Pulse Priors") for the regularisation hyperparameter,
which prove extremely flexible and complement empirical Bayes approaches,
such as cross-validation and maximum marginal likelihood, extremely well. This choice of prior is favourable to the Exponential hyperprior suggested 
in Park & Casella (2008).

These simulations are presented within the repository in the form of Jupyter Notebooks. Importantly, the BayesianLasso.py file contains a BayesianLasso class
that provides methods to compute posterior distributions with or without a hyperprior on the squared regularisation hyperparameter, as well as a method that performs coordinate
descent to train frequentist Lasso models. 

I will continue to update this repository with more simulations. One avenue that I would like to explore is the high dimensional dataset (p > n) case. There is also some potential
in investigating various mode finding methods for the problem.
