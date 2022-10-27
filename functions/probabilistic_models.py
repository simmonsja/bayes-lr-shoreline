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
    coeff1 = numpyro.sample("b",dist.Normal(0.5, 0.2))
    intercept = numpyro.sample("a",dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(2))

    mu = coeff1 * energy + intercept
    numpyro.deterministic("mu", mu)

    numpyro.sample("dshl", dist.Normal(mu, sigma), obs=dshl)

###############################################################################
###############################################################################
