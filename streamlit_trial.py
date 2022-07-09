import numpy as np
import pandas as pd
import streamlit as st

from functions import (calculate_wave_energy, find_pareto_front, linear_model,
                       plot_regression)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')
import arviz as az

#numpyro
from jax import random
import jax.numpy as jnp
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive

st.title("Storm Erosion Curve Fit")

c29, c30, c31 = st.columns([1, 6, 1])

# Initialization
if 'x_log' not in st.session_state:
    st.session_state['x_log'] = None
if 'y_log' not in st.session_state:
    st.session_state['y_log'] = None
if 'samples' not in st.session_state:
    st.session_state['samples'] = None

with c30:

    # load wave data
    rawWaveData = pd.read_csv('combined_era_data_-34.0_151.5.csv',index_col=0,parse_dates=True)
    # shoreline data
    transectName = 'aus0206-0005'
    csSource = 'http://coastsat.wrl.unsw.edu.au/time-series/{}/'
    tmpLoc =  "working_data.csv"
    # download = False

    # if download:
    #     urllib.request.urlretrieve(csSource.format(transectName), tmpLoc)

    rawShlData = pd.read_csv(tmpLoc,parse_dates=True,index_col=0,header=None)
    rawShlData.index = pd.to_datetime(rawShlData.index,utc=True)
    rawShlData.columns = ['Shoreline']
    rawShlData.index.name = 'Date'

    diffShlData = rawShlData.diff()
    diffShlData.columns = ['dShl'] 

    # st.pyplot(fig=sns.histplot(diffShlData.dropna()))

    # Prepare the diffShlData df
    shlData = rawShlData.diff().copy()
    shlData['postDate'] = shlData.index
    shlData.loc[shlData.index[1:],'preDate'] = shlData.index[:-1]
    shlData.drop(shlData.index[0],inplace=True)
    shlData.index = ['Storm_{0:04.0f}'.format(_) for _ in range(shlData.shape[0])]

    # split the wave data according to the shoreline data
    waveData = {}
    for thisStorm in shlData.index:
        waveData[thisStorm] = rawWaveData.loc[shlData.loc[thisStorm,'preDate']:shlData.loc[thisStorm,'postDate']]
        shlData.loc[thisStorm,'E'] = calculate_wave_energy(waveData[thisStorm])

    # constrain shlData by adding consective shoreline movements
    # print(shlData)
    shlData['zeroCross'] = np.sign(shlData['Shoreline']).diff().ne(0).astype(int)
    shlData['zeroCross'] = shlData['zeroCross'].cumsum()
    groupedVals = shlData.groupby(by='zeroCross',as_index=False).sum()
    groupedVals.index = shlData.index[[np.where(shlData['zeroCross'] == _)[0][0] for _ in groupedVals['zeroCross']]]
    shlData['E'] = groupedVals['E']
    shlData['Shoreline'] = groupedVals['Shoreline']
    shlData.drop_duplicates(subset=['zeroCross'],inplace=True)

    # paretoThresh = 0.75
    with st.form(key="pareto_clean"):
        paretoThresh = st.number_input(
            'paretoThresh:',
            value=0.75
        )

        submitted = st.form_submit_button("Run")

        if submitted:
            plotData = shlData.dropna().copy()
            plotData.loc[:,'Shoreline'] = -plotData['Shoreline']
            # some basic reduction of clearly dodgy data
            plotData = plotData.loc[(plotData['E']>0.25e6)&(plotData['Shoreline']>1),:]
            plotData = find_pareto_front(plotData)

            cleanData = plotData.copy()
            cleanData = cleanData.loc[cleanData['paretoDistance']<paretoThresh,:]
            x, y = cleanData['E'].values, cleanData['Shoreline'].values

            # and trasnform to logspace
            x_log = np.log(x)
            y_log = np.log(y)

            st.session_state.x_log = x_log
            st.session_state.y_log = y_log

            log_scale=False
            fig = plt.figure(figsize=(7, 5))
            ax1 = fig.add_subplot(111)

            sns.scatterplot(x='E',y='Shoreline',hue='paretoDistance',data=plotData,ax=ax1)
            sns.scatterplot(x='E',y='Shoreline',color='C2',marker='x',data=plotData.loc[plotData['paretoDistance']<paretoThresh,:],ax=ax1)
            sns.scatterplot(x='E',y='Shoreline',color='C0',marker='s',data=plotData.loc[plotData['pareto'].astype(bool),:],ax=ax1)

            ax1.set_xlabel('E')
            ax1.set_ylabel('dShl')
            # ax1.invert_yaxis()
            # change axes to log - log scale
            if log_scale:
                plt.xscale('log')
                plt.yscale('log')
            plt.title('Shoreline change from satelite')
            ax1.get_legend().remove()
            plt.legend(loc='center', bbox_to_anchor=(1.15,0.5))
            st.pyplot(fig=plt.gcf())
    
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
            mcmc_obj.print_summary()
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
            fig, ax = plt.subplots(3,2,figsize=(10, 7))
            az.plot_trace(numpyro_data,axes=ax)
            st.pyplot(fig=fig)
            
            x_log_out = np.linspace(x_log.min(),x_log.max()+(x_log.max()-x_log.min())*0.25,100)

            # Compute empirical posterior distribution over mu
            posterior_mu = (
                jnp.expand_dims(samples["intercept"], -1)
                + jnp.expand_dims(samples["coeff1"], -1) * x_log_out
            )

            mean_mu = jnp.mean(posterior_mu, axis=0)
            hpdi_mu = hpdi(posterior_mu, ci)
            ax = plot_regression(np.exp(x_log), np.exp(y_log), np.exp(x_log_out), np.exp(mean_mu), np.exp(hpdi_mu), log_scale=True)
            ax.set(
                xlabel="Energy", ylabel="dShl", title="Regression line with 95% CI"
            )
            st.pyplot(fig=ax.get_figure())
            ax = plot_regression(np.exp(x_log), np.exp(y_log), np.exp(x_log_out), np.exp(mean_mu), np.exp(hpdi_mu))
            ax.set(
                xlabel="Energy", ylabel="dShl", title="Regression line with 95% CI"
            )
            st.pyplot(fig=ax.get_figure())

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