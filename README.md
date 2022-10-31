# Bayesian Linear Regression

Written by Joshua Simmons 2022

In this notebook, we will fit a Bayesian Linear Regression to predict shoreline change due to coastal storms. This will mirror the simple empirical model developed by:

*Harley, M. D., Turner, I. L., Short, A. D., & Ranasinghe, R. (2009). An empirical model of beach response to storms–SE Australia. In Coasts and Ports (pp. 600-606).*

[Paper available here.](https://www.researchgate.net/profile/Mitchell-Harley/publication/267506992_An_empirical_model_of_beach_response_to_storms_-_SE_Australia/links/5cd8eb9f458515712ea6801f/An-empirical-model-of-beach-response-to-storms-SE-Australia.pdf)

This model is of the form: $\Delta W=aE^b$, where $\Delta W$ is the change in shoreline position, $E$ is the storm energy, and $a$ and $b$ are learnable model parameters.

To provide uncertainty alongside the model prediction, we will use the probabilistic programming language [NumPyro](https://numpyro.readthedocs.io/en/stable/) to fit a Bayesian Linear Regression.

Disclaimers:

- This is an overly simplified analysis for the purpose of demonstrating uncertainty quantification (via Bayesian inference). 
- It was developed purely for practice with NumPyro and streamlit.
- Model predicitons of shoreline change should not be relied upon for real-world application.
