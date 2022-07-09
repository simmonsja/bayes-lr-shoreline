import numpy as np
from scipy.spatial.distance import cdist

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
        if row['E'] > prevE and row['Shoreline'] > prevShl:
            data.loc[ind,'pareto'] = 1
            prevE = row['E']
            prevShl = row['Shoreline']
    # and then give distance to pareto front

    data['paretoDistance'] = 0
    paretoPoints = data.loc[data['pareto'] == 1,['E','Shoreline']].values
    eughPoints = data.loc[data['pareto'] == 0,['E','Shoreline']].values
    pDists = cdist(paretoPoints,eughPoints,'seuclidean').min(axis=0)
    data.loc[data['pareto'] == 0,'paretoDistance'] = pDists
    return data

###############################################################################
###############################################################################
