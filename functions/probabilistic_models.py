import numpyro
import numpyro.distributions as dist

###############################################################################
###############################################################################

def linear_model(energy=None,dshl=None):
    '''
    Define linear model with priors for the parameters and model error
    This is a log of the original form dshl = a*E^b
    Inputs:
        energy: storm energy
        dshl: observed shoreline change
    '''
    # Define priors
    a = numpyro.sample("a",dist.Normal(0, 10))
    b = numpyro.sample("b",dist.Normal(1, 0.5))

    sigma = numpyro.sample("sigma", dist.Exponential(1))
    mu = a + b * energy
    # store the model prediction before we account for the error
    numpyro.deterministic("mu", mu)
    # and then finally sample so we can compare to our observations
    dshl_modelled = numpyro.sample("dshl_modelled", dist.Normal(mu, sigma), obs=dshl)

###############################################################################
###############################################################################
