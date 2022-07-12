import os
import urllib.request

import pandas as pd

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
    tmpLoc =  "{}.csv".format(transect_name)

    if download and not os.path.exists(tmpLoc):
        urllib.request.urlretrieve(cs_source.format(transect_name), tmpLoc)

    raw_shl_data = pd.read_csv(tmpLoc,parse_dates=True,index_col=0,header=None)
    raw_shl_data.index = pd.to_datetime(raw_shl_data.index,utc=True)
    raw_shl_data.columns = ['Shoreline']
    raw_shl_data.index.name = 'Date'

    diff_shl_data = raw_shl_data.diff()
    diff_shl_data.columns = ['dShl']
    return diff_shl_data

###############################################################################
###############################################################################

def load_wave_data(transect_name='aus0206-0005'):
    '''
    Load the wave data - this should lookup the appropriate ERA5 
    collection from the transect name.
    '''
    raw_wave_data = pd.read_csv('combined_era_data_-34.0_151.5.csv',index_col=0,parse_dates=True)
    return raw_wave_data

###############################################################################
###############################################################################
