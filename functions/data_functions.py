import os
import urllib.request

import numpy as np
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
    # make sure we don't accumulate a tonne - I think i could just parse from the url
    # but lets just go with this for now
    os.remove(tmpLoc)
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
    # load the location data - geopandas was being a pain in docker
    # transect_loc = os.path.join(".","data","CoastSat_transect_layer.geojson")
    # # import geojson
    # transects = gpd.read_file(transect_loc,driver='GeoJSON')
    # au_transects = transects.query('SiteId.str.startswith("aus").values')
    # # extract the lat lon to a new column
    # au_transects.loc[:,'longitude'] = au_transects['geometry'].apply(lambda x: (x.xy[0][0]))
    # au_transects.loc[:,'latitutde'] = au_transects['geometry'].apply(lambda x: (x.xy[1][0]))
    # transects_out = au_transects[['SiteId','TransectId','Orientation','longitude','latitutde']]
    # transects_out.to_csv(os.path.join(".","data","CoastSat_transect_layer.csv"))
    
    # load the transect locations
    transect_loc = os.path.join(".","data","CoastSat_transect_layer.csv")
    transects = pd.read_csv(transect_loc,index_col=0)
    transects.set_index('TransectId',inplace=True)
    if not transect_name in transects.index:
        raise ValueError("Transect name not found - please select an Australian transect")
    lat = transects.loc[transect_name,'latitutde']
    # lon = transects.loc[transect_name,'longitude']

    wave_data_lat = np.array([-27.0,-28.0,-29.0,-30.0,-31.0,-32.0,-33.0,-34.0,-35.0,-36.0,-37.0])
    wave_data_lon = np.array([153.5,153.5,153.5,153.5,153.5,153.0,152.0,151.5,151.0,150.5,150.0])

    # find the closest by lat
    lat_idx = np.argmin(np.abs(wave_data_lat-lat))
    
    # load the wave data
    buoy_name = 'era5_{}_{}'.format(wave_data_lat[lat_idx],wave_data_lon[lat_idx])
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
