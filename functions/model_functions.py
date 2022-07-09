import numpyro
import numpyro.distributions as dist

###############################################################################
###############################################################################

def linear_model(energy=None,dshl=None):
    '''
    Define linear model with priors for the parameters and model error
    Need to check that the assumptions for priors give the right posterior
    '''
    # Define priors
    coeff1 = numpyro.sample("coeff1",dist.Normal(0, 10))
    intercept = numpyro.sample("intercept",dist.Normal(0, 10))
    sigma = numpyro.sample("sigma", dist.Gamma(2.5,5))

    mu_out = coeff1 * energy + intercept
    dshl_modelled = numpyro.sample("dshl_modelled", dist.Normal(mu_out, sigma), obs=dshl)

###############################################################################
###############################################################################
