import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


sns.set_style('whitegrid')
sns.set_context('talk')

###############################################################################
###############################################################################

def plot_regression(x, y_obs, x_mod, y_mean, y_hpdi, y_predci,log_scale=False):
    # Sort values for plotting by x axis
    idx = np.argsort(x)
    idx_mod = np.argsort(x_mod)
    marriage = x[idx]
    plot_x_mod = x_mod[idx_mod]
    mean = y_mean[idx_mod]
    hpdi = y_hpdi[:, idx_mod]
    divorce = y_obs[idx]
    predci = y_predci[:, idx_mod]

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        locmin = mticker.MultipleLocator(10)  
        ax.yaxis.set_major_locator(locmin)
        # ax.yaxis.set_major_formatter(mticker.LogFormatter())
        locmin = mticker.MultipleLocator(1e6)  
        ax.xaxis.set_major_locator(locmin)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter()) 
    ax.plot(plot_x_mod, mean)
    ax.plot(marriage, divorce, "o")
    ax.fill_between(plot_x_mod, predci[0], predci[1], alpha=0.2, color='C1', interpolate=True)
    ax.fill_between(plot_x_mod, hpdi[0], hpdi[1], alpha=0.4, color='C0', interpolate=True)

    return ax

###############################################################################
###############################################################################

def plot_pareto_points(plotData,hue='paretoDistance',pareto_thresh=None,log_scale=False):
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)

    sns.scatterplot(x='E',y='dShl',hue=hue,data=plotData,ax=ax1)
    if not pareto_thresh is None:
        sns.scatterplot(x='E',y='dShl',color='C1',data=plotData.loc[plotData['paretoDistance']<pareto_thresh,:],ax=ax1)
        sns.scatterplot(x='E',y='dShl',color='C0',marker='s',data=plotData.loc[plotData['pareto'].astype(bool),:],ax=ax1)
    ax1.set_xlabel('E')
    ax1.set_ylabel('dShl')
    # ax1.invert_yaxis()
    # change axes to log - log scale
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.title('Shoreline change from satellite')
    ax1.get_legend().remove()
    plt.legend(loc='center', bbox_to_anchor=(1.15,0.5))
    return fig

###############################################################################
###############################################################################

def draw_fit(xObs,yObs,xPred,yPred,ySample=None,yMAP=None,**kwargs):
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(111)

    # if kwargs.get('errBands',False):
    #     ax1.fill_between(xSample, predyMean - 1.96*predyStd, predyMean + 1.96*predyStd, alpha=0.2,label='95% CI')
    #     ax1.plot(xSample, sampleyMean[:,0],label='Samples')
    #     ax1.plot(xSample, sampleyMean[:,1:])
    # else:
    #     ax1.plot(xSample, sampleyMean)#,label='Samples')

    if not ySample is None:
        ax1.plot(xPred, ySample[0,:].T,'-',color='xkcd:light grey',alpha=0.4,label='Samples')
        ax1.plot(xPred, ySample[1:,:].T,'-',color='xkcd:light grey',alpha=0.4)

    if not yMAP is None:
        ax1.plot(xPred, yMAP.T,'-',color='xkcd:red',label='MAP fit')

    # # plot mean
    # ax1.plot(xSample, predyMean, 'k', label='Mean')

    # if not y is None:
    ax1.plot(xObs, yObs, 'o',color='xkcd:dark grey',label='Observed')
    ax1.plot(xPred, yPred, 'C0', label='Predicted')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    if kwargs.get('log_scale',False):
        ax1.set_xscale('log')
        ax1.set_yscale('log')

    ax1.set_title(kwargs.get('title','Power Law Distribution Fit'))
    # place legend outside the plot
    plt.legend(loc='center', bbox_to_anchor=(1.2, 0.5))
    plt.show()
    return ax1
    
###############################################################################
###############################################################################
