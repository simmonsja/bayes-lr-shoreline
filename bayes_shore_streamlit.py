import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import arviz as az
import streamlit as st
import copy

sns.set_style('whitegrid')
sns.set_context('talk')

# plotting
from functions import (plot_regression, plot_pareto_points, draw_fit) 
# data
from functions import(load_shoreline_data, load_wave_data, clean_dshoreline_data) 
# analysis
from functions import (find_pareto_front, generate_storm_dataset, calculate_storm_energy)
# models
from functions import (linear_model)

# NumPyro for proabilistic programming
from jax import random
import jax.numpy as jnp
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive

st.set_page_config(
    page_title="Storm Erosion Model",
    page_icon="🌊",
    layout="wide"
)

# def update_width():
#     '''From Streamlit examples'''
#     max_width_str = "max-width: 1400px;"
#     st.markdown(
#         '''
#         <style>
#         .main .block-container{{
#             {}
#         }}
#         </style>    
#         '''.format(max_width_str),
#         unsafe_allow_html=True,
#     )
# update_width()

st.title("Storm Erosion Model with Uncertainty")
st.write('For a more detailed explanation, please see the associated [Github repo](https://github.com/simmonsja/bayes-lr-shoreline). Expand the sections below to view the outputs.')
with st.expander('Model Explanation'):
    st.write(
        '''
        In this streamlit, we will fit a Bayesian Linear Regression to predict shoreline change due to coastal storms. 
        
        This will mirror the simple empirical model developed by Harley et al. (2009) of the form: 

        $$\Delta W=aE^b$$
        
        where $\Delta W$ is the change in shoreline position, $E$ is the storm energy, 
        and $a$ and $b$ are learnable model parameters.
        
        To provide uncertainty alongside the model prediction, we will use the probabilistic programming language
        [NumPyro](https://numpyro.readthedocs.io/en/stable/) to fit a Bayesian Linear Regression.

        Disclaimers:

        - This is an overly simplified analysis for the purpose of demonstrating uncertainty quantification (via Bayesian inference). 
        - It was developed purely for practice with NumPyro and streamlit.
        - Model predicitons of shoreline change should not be relied upon for real-world application.
        '''
    )


_, cInteract, cOutput = st.columns([1, 2, 1])

# Initialize some states which will flow through
if 'x_log' not in st.session_state:
    st.session_state['x_log'] = None
if 'y_log' not in st.session_state:
    st.session_state['y_log'] = None
if 'storm_thresh' not in st.session_state:
    st.session_state['storm_thresh'] = 3.0
if 'samples' not in st.session_state:
    st.session_state['samples'] = None
if 'data_fig' not in st.session_state:
    st.session_state['data_fig'] = None
if 'pareto_fig' not in st.session_state:
    st.session_state['pareto_fig'] = None
if 'regression_fig 'not in st.session_state:
    st.session_state['regression_fig'] = None
if 'params_fig' not in st.session_state:
    st.session_state['params_fig'] = None

# Load data - group these calls together
@st.experimental_singleton
def load_data(t_name):
    # sped up loading of data that caches
    raw_shl_data = load_shoreline_data(transect_name=t_name)
    raw_wave_data, t_loc = load_wave_data(transect_name=t_name)
    shl_data, storm_thresh = generate_storm_dataset(raw_shl_data, raw_wave_data)
    return shl_data, storm_thresh, t_loc

with cInteract:
    st.subheader("Load and clean the data...")
    st.write(
        '''
        As this is a simplified analysis, we will use satellite data derived using 
        [CoastSat](https://github.com/kvos/CoastSat). Specifically, we will use data 
        from this great resource ([the CoastSat website](http://coastsat.wrl.unsw.edu.au))
        which provides preprocessed shorelines for most of the East Coast of Australia.

        Use the [CoastSat website](http://coastsat.wrl.unsw.edu.au) to select a transect of interest 
        and copy the transect ID to below. 
        '''
    )
    transect_name = st.text_input(
        "Transect ID:",
        "aus0206-0005"
    )
    shl_data, st.session_state['storm_thresh'], transect_loc = load_data(transect_name)
    
    st.map(
        pd.DataFrame(
            np.array(transect_loc).reshape(1,-1),
            columns=['lat', 'lon']
        )
    )

    # now we will plot the points for cleaning
    st.session_state['data_fig'] = plot_pareto_points(shl_data,hue='timeDelta')

    with st.expander('Raw data plot', expanded=True):
        st.write('All data points:')
        if not st.session_state['data_fig'] is None:
            st.pyplot(st.session_state['data_fig']) 

    with st.form(key="pareto_clean"):
        st.write('Here we will use some thresholds to clean the data.')
        with st.expander('Data cleaning'):
            st.write(
                '''
                To make life easier for ourselves, we will look for the Pareto front which defines the 
                upper envelope of observed shoreline change for a given storm energy. 
                We will then use some thresholds to remove all storms which are either too far 
                from the Pareto front, or otherwise deemed unacceptable.

                We will need to choose:

                - Pareto threshold: the threshold for distance from the Pareto front 
                to be considered a valid storm event
                - Time threshold: the maximum time between pre- and post- storm surveys 
                to avoid problems due to low availability of satellite data
                - Energy threshold: the minimum energy to be considered a valid event
                '''
            )
        pareto_thresh = st.number_input(
            'Pareto Threshold (euclidean distance to the pareto front):',
            value=0.7
        )
        time_thresh = st.number_input(
            'Maximum days between shoreline measurements:',
            value=180
        )

        energy_thresh = st.number_input(
            'Minimum event energy:',
            value=0.25e6
        ) 

        submitted = st.form_submit_button("Run")

        if submitted:
            x, y, plot_data = clean_dshoreline_data(shl_data, pareto_thresh, time_thresh, energy_thresh)

            # and transform to logspace and store for later
            st.session_state.x_log = np.log(x)
            st.session_state.y_log = np.log(y)
            st.session_state['pareto_fig'] = plot_pareto_points(plot_data,pareto_thresh=pareto_thresh)
            # st.pyplot(fig=pareto_fig)

    with st.expander('Selected data plot', expanded=True):
        if not st.session_state['pareto_fig'] is None:
            st.write('Selected points based on thresholds:')
            st.pyplot(st.session_state['pareto_fig'])

    st.subheader("Fit a Bayesian Linear Regression model...") 
    with st.expander('Model details'):
        st.write(
            '''
            We will fit our Bayesian Linear Regression using NumPyro. Our model looks similar to before 
            (after taking logs to turn this into a linear regression):

            $$\log(\Delta W)=\log(a)+b*\log(E) + \epsilon$$

            $$\epsilon \sim \mathcal{N}(0,\sigma)$$

            but here we have explicitly stated that the model fits the data with some 
            random residual or error ($\epsilon$). We can then model this error when fitting to the data, 
            to take into account uncertainty.

            We will use the package [NumPyro](https://num.pyro.ai/en/stable/getting_started.html) to sample 
            and determine the paramaters in our model ($log(a)$, $b$, and $sigma$). For an introduction to Bayesian methods, 
            I thoroughly recommend Richard McElreath's [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) book.

            If you want, you can experiment with the number of samples to get a better estimate of the 
            posterior distribution (or see how poor the description of the posterior becomes if we take less samples).
            '''
        )
    with st.expander('Uncertainty estimate'):
        st.write(
            '''
            We can plot our model predictions with uncertainty for our selected credible interval - say 68%.

            Notice that we plot two uncertainty bands, the first is the credible interval for our linear model (blue bands). 
            Essentially, what are range of slopes and intercepts that linear fit could take, reflecting uncertainty in
            parameters $a$ and $b$ as we are uncertain of their exact value. 

            However, we also have the $\sigma$ parameter. The predictive uncertainty (credible interval given by the orange bands) 
            incorporates model uncertainty and the error ($\epsilon$) which describes our data. Here we are simulating possible 
            values of actual data points, as opposed to just the range of possible linear 
            regression lines. This range reflects the shoreline change we could expect for a new event if measured by Coastsat data, 
            rather than our modelled prediction of actual shoreline change. These bands should include approximately 68% of our data 
            points.
            '''
        )

    with st.form(key="model_fit"):
        # settings 
        num_samples = st.number_input(
            'Number of MCMC samples:', 
            value=1000, 
            min_value=100, 
            max_value=20000,
            step=100
        )
        burnin = int(0.25 * num_samples)
        ci = st.number_input(
            'Credible interval (in %):', 
            value=68, 
            min_value=50, 
            max_value=99,
            step=1
        )/100

        # Random number generator - jax style
        rng_key = random.PRNGKey(2101)
        rng_key, rng_key_ = random.split(rng_key)
        x_log = st.session_state.x_log
        y_log = st.session_state.y_log

        # define the sampler - No U-Turn Sampler (NUTS)
        kernel = NUTS(linear_model)

        submitted_model = st.form_submit_button("Run")

        if submitted_model:
            # define the mcmc wrapper  
            mcmc_obj = MCMC(kernel, num_warmup=burnin, num_samples=num_samples)
            with st.spinner('Please wait while the MCMC sampler runs...'):
                mcmc_obj.run(
                    rng_key_, energy=x_log, dshl=y_log
                )
            st.success('MCMC sampling done!')
            # print('MCMC Summary - check for Rhat = 1:')
            # mcmc_summary = mcmc_obj.print_summary()
            # st.write(mcmc_summary)
            samples = mcmc_obj.get_samples()

            st.session_state['samples'] = samples

            # get the samples for predictive uncertainty (our linear model + error)
            posterior_predictive = Predictive(linear_model, samples)(
                rng_key_, energy=x_log)

            # get the mean model prediciton
            mean_mu = jnp.mean(posterior_predictive['mu'], axis=0)
            # hpdi is used to compute the credible intervals corresponding to ci
            hpdi_mu = hpdi(posterior_predictive['mu'], ci)
            hpdi_sim_y = hpdi(posterior_predictive['dshl_modelled'], ci)

            arviz_posterior = az.from_numpyro(
                mcmc_obj,
                posterior_predictive=posterior_predictive,
            )
        
            # Now we will plot the results in log-log scale and on the original scale
            log_reg_fig = plot_regression(np.exp(x_log), np.exp(y_log), np.exp(x_log), np.exp(mean_mu), np.exp(hpdi_mu), np.exp(hpdi_sim_y), ci, log_scale=True)
            reg_fig = plot_regression(np.exp(x_log), np.exp(y_log), np.exp(x_log), np.exp(mean_mu), np.exp(hpdi_mu), np.exp(hpdi_sim_y), ci)

            st.session_state['regression_fig'] = reg_fig

            # Plot the parameter traces
            params_ax = az.plot_trace(
                arviz_posterior,
                var_names=['a','b','sigma'],
                figsize=(8, 6),
            )
            plt.subplots_adjust(
                hspace=0.5,
                wspace=0.2
            )

            st.session_state['params_fig'] = plt.gcf()

        with st.expander('Model fit plot', expanded=True):
            if not st.session_state['regression_fig'] is None:
                st.write('Model fit with uncertainty:')
                st.pyplot(st.session_state['regression_fig'])
        with st.expander('Posterior parameter distribution plots'):
            if not st.session_state['params_fig'] is None:
                st.write(
                    '''
                    If you're interested in checking the MCMC sampling, you can see the 
                    traces of the parameters here. The parameter distrbituions give us an idea
                    of the uncertainty in our three model parameters. Check the Github repo for more
                    resources on Bayesian inference.
                    '''
                )
                st.pyplot(st.session_state['params_fig'])

    st.subheader("Predict shoreline change: storm energy...")
    st.write(
        '''
        Now we can use our model to predict the shoreline change for a given storm energy.
        '''
    )
    with st.form(key='predict'):
        samples = st.session_state['samples']

        x_in = st.number_input(
            'Event energy (Jh/m^2):',
            value=0.8e6,
            step=25000.0
        )

        submitted_energy = st.form_submit_button("Run")

        if submitted_energy:
            # make a prediction with our model
            event_predictive = Predictive(linear_model, samples)(
                rng_key_, energy=np.log(np.array(x_in)))

            st.write('Storm Energy: {:.2E}'.format(x_in))
            st.write('Model Mean predicted shoreline change: {:.2f} m'.format(np.exp(jnp.mean(event_predictive['mu']))))
            st.write('Model {:.0f}% Credibilty Interval: {:.2f} - {:.2f} m'.format(ci*100,*np.exp(hpdi(event_predictive['mu'], ci))))
            st.write('Prediction {:.0f}% Credibilty Interval: {:.2f} - {:.2f} m'.format(ci*100,*np.exp(hpdi(event_predictive['dshl_modelled'], ci))))

        
    st.subheader("Predict shoreline change: storm characteristics...")
    st.write(
        '''
        Without a known storm energy, it may be easier to assume a triangular storm, providing 
        a maxiumum $H_{sig}$ for the event and associated storm length (in hours). 
        We will assume the initial wave height for a storm is the 95% exceedance threshold for $H_{sig}$
        (about $3$ m for Sydney). The triangular storm increases in wave height to peak halfway through 
        the event with the specified maximum $H_{sig}$.
        '''
    )

    with st.form(key='predict_hsig'):
        samples = st.session_state['samples']

        hsig_in = st.number_input(
            'Hsig (m) maximum:',
            value=6.0
        )
        dur_in = st.number_input(
            'Storm duration (hours):',
            value=48
        )

        submitted_hsig = st.form_submit_button("Run")

        if submitted_hsig:
            event_energy = calculate_storm_energy(hsig_in, dur_in, storm_thresh=st.session_state['storm_thresh'])

            # make a prediction with our model
            event_predictive = Predictive(linear_model, samples)(
                rng_key_, energy=np.log(np.array(event_energy)))

            st.write('Storm Energy: {:.2E} with hsig_max = {:.2f} m and storm_duration = {:.0f} hrs'.format(event_energy, hsig_in, dur_in))
            st.write('Model Mean predicted shoreline change: {:.2f} m'.format(np.exp(jnp.mean(event_predictive['mu']))))
            st.write('Model {:.0f}% Credibilty Interval: {:.2f} - {:.2f} m'.format(ci*100,*np.exp(hpdi(event_predictive['mu'], ci))))
            st.write('Prediction {:.0f}% Credibilty Interval: {:.2f} - {:.2f} m'.format(ci*100,*np.exp(hpdi(event_predictive['dshl_modelled'], ci))))

    # download model - maybe I can implement this later on
    # st.download_button(
    #     label="Download model",
    #     data=,
    #     file_name="{}_lr_model.pkl".format(transect_name)
    # )