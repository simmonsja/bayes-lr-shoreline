import os
import urllib.request

import pandas as pd

from functions.analysis_functions import find_pareto_front

###############################################################################
###############################################################################

def load_shoreline_data(transect_name='aus0206-0005',download=True):
    '''
    This loads from coastsat but you could change this to 
    load any form of shoreline data you'd like
    Downloads pre-processed data from:
    http://coastsat.wrl.unsw.edu.au/
    '''
    cs_source = 'http://coastsat.wrl.unsw.edu.au/time-series/{}/'
    tmpLoc =  os.path.join(".","data","coastsat","{}.csv".format(transect_name))

    if download and not os.path.exists(tmpLoc):
        urllib.request.urlretrieve(cs_source.format(transect_name), tmpLoc)

    raw_shl_data = pd.read_csv(tmpLoc,parse_dates=True,index_col=0,header=None)
    raw_shl_data.index = pd.to_datetime(raw_shl_data.index,utc=True)
    raw_shl_data.columns = ['Shoreline']
    raw_shl_data.index.name = 'Date'
    return raw_shl_data

###############################################################################
###############################################################################

def load_wave_data(transect_name='aus0206-0005'):
    '''
    Load the wave data - this should lookup the appropriate ERA5 
    collection from the transect name.
    '''
    # load the wave data
    buoy_name = 'combined_era_data_-34.0_151.5'
    data_loc = os.path.join(".","data","waves","{}.csv".format(buoy_name))
    raw_wave_data = pd.read_csv(data_loc,index_col=0,parse_dates=True)
    return raw_wave_data

###############################################################################
###############################################################################

def clean_dshoreline_data(data_in,paretoThresh=0.75,timeThresh=150,energyThresh=0.25e6):
    plotData = data_in.dropna().copy()
    # some basic reduction of clearly dodgy data
    cleanBool = (plotData['E']>energyThresh)&(plotData['dShl']>1)&(plotData['timeDelta']<timeThresh)
    plotData = plotData.loc[cleanBool,:]
    # get the pareto front
    plotData = find_pareto_front(plotData)
    # now clean based on pareto distance
    cleanData = plotData.loc[plotData['paretoDistance']<paretoThresh,:].copy()
    x, y = cleanData['E'].values, cleanData['dShl'].values
    return x, y, plotData


###############################################################################
###############################################################################
