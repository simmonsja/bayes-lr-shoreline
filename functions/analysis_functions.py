import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

###############################################################################
###############################################################################

def generate_storm_dataset(shl_data,wave_data):
    '''
    Take in the shoreline and wave data and generate a dataset of 
    shoreline movement per storm.
    We are a little bit cheeky here and group consecutive
    shoreline movements together with the hope of getting a more 
    representative storm envelope.
    '''
    # Prepare the diffstorm_data df
    storm_data = shl_data.copy()

    storm_data = storm_data.diff()
    storm_data.columns = ['dShl']
    # define dShl as positive for recession
    storm_data.loc[:,'dShl'] = -storm_data['dShl']

    storm_data['postDate'] = storm_data.index
    storm_data.loc[storm_data.index[1:],'preDate'] = storm_data.index[:-1]
    storm_data.drop(storm_data.index[0],inplace=True)
    storm_data.index = ['Storm_{0:04.0f}'.format(_) for _ in range(storm_data.shape[0])]

    # split the wave data according to the shoreline data
    waveData = {}
    for thisStorm in storm_data.index:
        waveData[thisStorm] = wave_data.loc[storm_data.loc[thisStorm,'preDate']:storm_data.loc[thisStorm,'postDate']]
        storm_data.loc[thisStorm,'E'] = calculate_wave_energy(waveData[thisStorm])

    # constrain storm_data by adding consective shoreline movements
    storm_data['timeDelta'] = (storm_data['postDate'] - storm_data['preDate']).dt.days
    storm_data['zeroCross'] = np.sign(storm_data['dShl']).diff().ne(0).astype(int)
    storm_data['zeroCross'] = storm_data['zeroCross'].cumsum()
    groupedVals = storm_data.groupby(by='zeroCross',as_index=False).sum()
    groupedVals.index = storm_data.index[[np.where(storm_data['zeroCross'] == _)[0][0] for _ in groupedVals['zeroCross']]]
    storm_data['E'] = groupedVals['E']
    storm_data['dShl'] = groupedVals['dShl']
    storm_data['timeDelta'] = groupedVals['timeDelta']
    storm_data.drop_duplicates(subset=['zeroCross'],inplace=True)

    return storm_data


###############################################################################
###############################################################################

def calculate_wave_energy(waveIn):
    '''
    Calculate the wave energy for a storm - see Harley et al. 2009
    '''
    rho = 1025 # kg/m^3
    g = 9.81 # m/s^2
    # this needs to be more robust
    dt = (waveIn.index[1] - waveIn.index[0]).seconds/3600
    stormCriteria = waveIn['Hsig']>3.0 #m
    energy = np.sum(waveIn.loc[stormCriteria,'Hsig']**2)*rho*g*dt*(1/16)
    return energy


###############################################################################
###############################################################################

def calculate_storm_energy(hsig_max, storm_duration):
    '''
    Using this, the energy will be calculated for a triangular storm event. 
    We will assume the wave height to be defined as a storm is $H_{sig} = 3$ m 
    and the storm will increase in wave height to peak halfway through the event 
    with the specified maximum $H_{sig}$.
    '''
    storm_duration = int(storm_duration)
    dummy_time = pd.to_datetime('2022-01-01 00:00:00')
    dummy_delta = pd.to_timedelta(storm_duration,unit='H')
    times = pd.date_range(dummy_time,dummy_time+dummy_delta,freq='H')
    #simple triangular storm
    uptick = np.floor(storm_duration/2).astype(int)
    waves = np.zeros((storm_duration,))
    waves[:uptick] = np.linspace(3,hsig_max,uptick)
    waves[uptick:] = np.linspace(hsig_max,3,storm_duration-uptick)

    wave_data = pd.DataFrame(
        {'Hsig':waves},
        index=times[:-1]
    )
    # now use our handy energy calc above
    dummy_energy = calculate_wave_energy(wave_data)
    return dummy_energy

###############################################################################
###############################################################################

def find_pareto_front(data):
    '''
    Find the pareto front of a dataframe - we have a simple 2D 
    case where we know the variables so brute force it the 
    hard coded lazy way (:
    '''
    data['pareto'] = 0
    prevE = 0
    prevShl = -1e9
    data.sort_values(by='E',ascending=True,inplace=True)
    # roll on through and mark the pareto front
    for ind, row in data.iterrows():
        if row['E'] > prevE and row['dShl'] > prevShl:
            data.loc[ind,'pareto'] = 1
            prevE = row['E']
            prevShl = row['dShl']
    # and then give distance to pareto front

    data['paretoDistance'] = 0
    paretoPoints = data.loc[data['pareto'] == 1,['E','dShl']].values
    eughPoints = data.loc[data['pareto'] == 0,['E','dShl']].values
    pDists = cdist(paretoPoints,eughPoints,'seuclidean').min(axis=0)
    data.loc[data['pareto'] == 0,'paretoDistance'] = pDists
    return data

###############################################################################
###############################################################################
