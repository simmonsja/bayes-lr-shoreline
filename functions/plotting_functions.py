import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def plot_regression(x, y_obs, x_mod, y_mean, y_hpdi,log_scale=False):
    # Sort values for plotting by x axis
    idx = np.argsort(x)
    idx_mod = np.argsort(x_mod)
    marriage = x[idx]
    plot_x_mod = x_mod[idx_mod]
    mean = y_mean[idx_mod]
    hpdi = y_hpdi[:, idx_mod]
    divorce = y_obs[idx]

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
    ax.fill_between(plot_x_mod, hpdi[0], hpdi[1], alpha=0.3, interpolate=True)

    return ax
