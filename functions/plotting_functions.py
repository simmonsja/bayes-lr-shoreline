from tkinter import font
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches

sns.set(font_scale=1.2)
sns.set_style('whitegrid')
sns.set_context('talk')

###############################################################################
###############################################################################

def plot_regression(x, y_obs, x_mod, y_mean, y_hpdi, y_predci, ci, log_scale=False, **kwargs):
    '''
    Bit of a messy function - could do with a cleaning of the inputs.
    '''
    # Sort values for plotting by x axis
    sns.set(font_scale=1.2)
    sns.set_style('whitegrid')

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        locmin = mticker.MultipleLocator(10)  
        ax.yaxis.set_major_locator(locmin)
        # ax.yaxis.set_major_formatter(mticker.LogFormatter())
        locmin = mticker.MultipleLocator(1e6)  
        ax.xaxis.set_major_locator(locmin)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter()) 
    ax.plot(x_mod, y_mean)
    ax.plot(x, y_obs, "o")
    ax.fill_between(
        x_mod, y_predci[0], y_predci[1], 
        alpha=0.2, color='C1', interpolate=True,
        label="Prediction\nuncertainty"
    )
    ax.fill_between(
        x_mod, y_hpdi[0], y_hpdi[1],
        alpha=0.4, color='C0', interpolate=True,
        label="Model\nuncertainty"
    )

    ax.set_xlabel(kwargs.get('xlabel', 'Energy'), labelpad=10)
    ax.set_ylabel(kwargs.get('ylabel', 'Shoreline Change'), labelpad=10)
    ax.set_title(kwargs.get('title', 'Regression line with {}% CI'.format(int(ci * 100))), pad=15)

    ax.legend(loc='center left', bbox_to_anchor=(1.05,0.5))

###############################################################################
###############################################################################

def plot_pareto_points(plotData,hue='paretoDistance',pareto_thresh=None,log_scale=False):
    sns.set(font_scale=1.2)
    sns.set_style('whitegrid')
    sns.set_context('talk')

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)

    sns.scatterplot(x='E',y='dShl',hue=hue,data=plotData,ax=ax1)
    if not pareto_thresh is None:
        selax = sns.scatterplot(x='E',y='dShl',color='C2',data=plotData.loc[plotData['paretoDistance']<pareto_thresh,:],ax=ax1)
        pfax = sns.scatterplot(x='E',y='dShl',color='C0',marker='s',data=plotData.loc[plotData['pareto'].astype(bool),:],ax=ax1)
    ax1.set_xlabel('E',labelpad=10)
    ax1.set_ylabel('dShl',labelpad=10)
    # ax1.invert_yaxis()
    # change axes to log - log scale
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.title('Shoreline change from coastsat data', pad=15)
    handles, labels = ax1.get_legend_handles_labels()
    selhandles, _ = selax.get_legend_handles_labels()
    pfhandles, _ = pfax.get_legend_handles_labels()

    selhandles = mpatches.Patch(color='C2', label='Selected')
    pfhandles = mpatches.Patch(color='C0', label='Pareto Front')

    plt.legend(
        handles = handles + [selhandles,pfhandles],
        labels = labels + ['Selected points','Pareto front'],
        loc='center left', bbox_to_anchor=(1.05,0.5),
        title='Pareto Distance')

    return fig

###############################################################################
###############################################################################

def draw_fit(x_obs,y_obs,x_pred,y_pred = None,y_sample=None,**kwargs):
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)

    # if kwargs.get('errBands',False):
    #     ax1.fill_between(xSample, predyMean - 1.96*predyStd, predyMean + 1.96*predyStd, alpha=0.2,label='95% CI')
    #     ax1.plot(xSample, sampleyMean[:,0],label='Samples')
    #     ax1.plot(xSample, sampleyMean[:,1:])
    # else:
    #     ax1.plot(xSample, sampleyMean)#,label='Samples')

    if not y_sample is None:
        ax1.plot(x_pred, y_sample[0,:].T,'-',color='xkcd:light grey',alpha=0.4,label='Samples')
        ax1.plot(x_pred, y_sample[1:,:].T,'-',color='xkcd:light grey',alpha=0.4)


    # if not y is None:
    ax1.plot(x_obs, y_obs, 'o',color='xkcd:dark grey',label='Observed')

    if not y_pred is None:
        ax1.plot(x_pred, y_pred, 'C0', label='Predicted')

    # # plot mean
    # ax1.plot(xSample, predyMean, 'k', label='Mean')

    
    
    ax1.set_xlabel(kwargs.get('xlabel','Energy'),labelpad=10)
    ax1.set_ylabel(kwargs.get('ylabel','Shoreline Change'),labelpad=10)

    if kwargs.get('log_scale',False):
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    ax1.set_title(kwargs.get('title','Power Law Distribution Fit'),pad=10)
    # place legend outside the plot
    plt.legend(loc='center', bbox_to_anchor=(1.2, 0.5))
    plt.show()
    return ax1
    
###############################################################################
###############################################################################
