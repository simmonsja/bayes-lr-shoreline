import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# plotting
from functions import (plot_regression, plot_pareto_points) 
# model 
from functions import (linear_model)
# data
from functions import(load_shoreline_data, load_wave_data) 
# analysis
from functions import (find_pareto_front,generate_storm_dataset)

sns.set_style('whitegrid')
sns.set_context('talk')
import arviz as az
import jax.numpy as jnp
#numpyro
from jax import random, transfer_guard_device_to_device
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive

st.set_page_config(
    page_title="Storm Erosion Curve Fit",
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

st.title("Storm Erosion Curve Fit")

cInteract, cOutput = st.columns([1, 3])

# Initialization
if 'x_log' not in st.session_state:
    st.session_state['x_log'] = None
if 'y_log' not in st.session_state:
    st.session_state['y_log'] = None
if 'samples' not in st.session_state:
    st.session_state['samples'] = None
if 'data_fig' not in st.session_state:
    st.session_state['data_fig'] = None
if 'pareto_fig' not in st.session_state:
    st.session_state['pareto_fig'] = None
if 'params_fig' not in st.session_state:
    st.session_state['params_fig'] = None
if 'regression_fig 'not in st.session_state:
    st.session_state['regression_fig'] = None

@st.experimental_singleton
def load_data(t_name):
    # sped up loading of data that caches
    raw_shl_data = load_shoreline_data(transect_name=t_name)
    raw_wave_data = load_wave_data(transect_name=t_name)
    shl_data = generate_storm_dataset(raw_shl_data, raw_wave_data)
    return shl_data

with cInteract:
    st.subheader("Load and clean the data...")
    transect_name = st.text_input(
        "Transect name:",
        "aus0206-0005"
    )
    shl_data = load_data(transect_name)
    
    st.session_state['data_fig'] = plot_pareto_points(shl_data,hue='timeDelta')
    # st.pyplot(fig=data_fig)

    # paretoThresh = 0.75
    with st.form(key="pareto_clean"):
        paretoThresh = st.number_input(
            'Pareto Threshold\n(seuclidean distance to the pareto front):',
            value=0.75
        )
        timeThresh = st.number_input(
            'Maximum days between shoreline:',
            value=180
        )

        eThresh = st.number_input(
            'Minimum energy:',
            value=0.25e6
        ) 

        submitted = st.form_submit_button("Run")

        if submitted:
            plotData = shl_data.dropna().copy()
            # some basic reduction of clearly dodgy data
            cleanBool = (plotData['E']>eThresh)&(plotData['dShl']>1)&(plotData['timeDelta']<timeThresh)
            plotData = plotData.loc[cleanBool,:]
            # get the pareto front
            plotData = find_pareto_front(plotData)
            # now clean based on pareto distance
            cleanData = plotData.loc[plotData['paretoDistance']<paretoThresh,:].copy()
            x, y = cleanData['E'].values, cleanData['dShl'].values

            # and transform to logspace and store for later
            st.session_state.x_log = np.log(x)
            st.session_state.y_log = np.log(y)

            log_scale=False
            st.session_state['pareto_fig'] = plot_pareto_points(plotData,pareto_thresh=paretoThresh)
            # st.pyplot(fig=pareto_fig)
    st.subheader("Fit a Bayesian Linear Regression model...") 
    with st.form(key="model_fit"):
        # settings 
        num_samples = st.number_input(
            'Number of MCMC samples:', 
            value=1000, 
            min_value=1000, 
            max_value=100000,
            step=1000
        )
        burnin = int(0.25 * num_samples)
        ci = 0.95

        # Random number generator - jax style
        rng_key = random.PRNGKey(2022)
        rng_key, rng_key_ = random.split(rng_key)
        x_log = st.session_state.x_log
        y_log = st.session_state.y_log

        # define the sampler
        kernel = NUTS(linear_model)

        submitted = st.form_submit_button("Run")

        if submitted:
            # define the mcmc wrapper
            mcmc_obj = MCMC(kernel, num_warmup=burnin, num_samples=num_samples)

            mcmc_obj.run(
                rng_key_, energy=x_log, dshl=y_log
            )
            print('MCMC Summary - check for Rhat = 1:')
            st.write(mcmc_obj.print_summary())
            samples = mcmc_obj.get_samples()

            st.session_state.samples = samples

            posterior_predictive = Predictive(linear_model, samples)(
                rng_key_, energy=x_log, dshl=y_log
            )
            # predictions =  posterior_predictive(rng_key_, energy=x_log)["dshl_modelled"]
            prior = Predictive(linear_model, num_samples=100)(
                rng_key_, energy=x_log, dshl=y_log
            )

            numpyro_data = az.from_numpyro(
                mcmc_obj,
                prior=prior,
                posterior_predictive=posterior_predictive,
                # coords={"school": np.arange(eight_school_data["J"])},
                # dims={"theta": ["school"]},
            )
            # fig = plt.figure(figsize=(10, 7))
            params_fig, ax = plt.subplots(
                3,2,
                figsize=(10, 7))
            plt.subplots_adjust(
                hspace=0.5,
                wspace=0.2
            )
            az.plot_trace(numpyro_data,axes=ax)
            st.session_state['params_fig'] = params_fig
            # st.pyplot(fig=params_fig)
            
            x_log_out = np.linspace(x_log.min(),x_log.max()+(x_log.max()-x_log.min())*0.25,100)

            # Compute empirical posterior distribution over mu
            posterior_mu = (
                jnp.expand_dims(samples["intercept"], -1)
                + jnp.expand_dims(samples["coeff1"], -1) * x_log_out
            )
            sim_dshl = Predictive(linear_model, samples)(
                rng_key_, energy=x_log_out
            )


            mean_mu = jnp.mean(posterior_mu, axis=0)
            hpdi_mu = hpdi(posterior_mu, ci)
            hdpi_sim = hpdi(sim_dshl, ci)
            ax = plot_regression(np.exp(x_log), np.exp(y_log), np.exp(x_log_out), np.exp(mean_mu), np.exp(hpdi_mu), np.exp(hdpi_sim), log_scale=True)
            ax.set(
                xlabel="Energy", ylabel="dShl", title="Regression line with 95% CI"
            )
            st.pyplot(fig=ax.get_figure())
            reg_ax = plot_regression(np.exp(x_log), np.exp(y_log), np.exp(x_log_out), np.exp(mean_mu), np.exp(hpdi_mu), np.exp(hdpi_sim))
            reg_ax.set(
                xlabel="Energy", ylabel="dShl", title="Regression line with 95% CI"
            )
            st.session_state['regression_fig'] = reg_ax.get_figure()
            # st.pyplot(fig=ax.get_figure())
    st.subheader("Predict for a new E...")
    with st.form(key='predict'):
        samples = st.session_state.samples

        x_in = st.number_input(
            'Event energy:',
            value=2e6
        )

        submitted = st.form_submit_button("Run")

        if submitted:
            x_log = np.log(x_in)
            # Compute empirical posterior distribution over mu
            posterior_mu = (
                jnp.expand_dims(samples["intercept"], -1)
                + jnp.expand_dims(samples["coeff1"], -1) * x_log
            )

            mean_mu = jnp.mean(posterior_mu, axis=0)
            hpdi_mu = hpdi(posterior_mu, ci)

            st.write(
                '''
                Predicted shoreline change: {:.2f} (m)\n
                95% credible interval: [{:.2f}, {:.2f}] (m)
                '''.format(
                    np.exp(mean_mu[0]),
                    np.exp(hpdi_mu[0][0]),
                    np.exp(hpdi_mu[1][0])
                )
            )
        
        # download model later
    st.download_button(
        label="Download model",
        data=,
        file_name="{}_lr_model.pkl".format(transect_name)
    )


with cOutput:
    if not st.session_state['data_fig'] is None:
        st.pyplot(st.session_state['data_fig'])
    if not st.session_state['pareto_fig'] is None:
        st.pyplot(st.session_state['pareto_fig'])
    if not st.session_state['params_fig'] is None:
        st.pyplot(st.session_state['params_fig'])
    if not st.session_state['regression_fig'] is None:
        st.pyplot(st.session_state['regression_fig'])