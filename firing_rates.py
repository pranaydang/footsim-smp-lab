import footsim as fs
import numpy as np
# import holoviews as hv
import holoviews as hv
hv.extension('mpl')
from footsim.plotting import plot

import warnings
warnings.filterwarnings('ignore')

def afferent_populations():

    forefoot = []
    hindfoot = []
    a = fs.affpop_foot()

    # Loop through the actual Afferent objects:
    for aff in a.afferents:
        # aff.location is a (1,2) array [[x, y]], so:
        aff.location[0, 0] = 0   # set x → 20

    # Now when you read the property, you’ll see that change:
    for i in range(len(a.afferents)):
        if a.afferents[i].location[0,1] < 10 and a.afferents[i].location[0,1] > -30:
            forefoot.append(a.afferents[i])

        elif a.afferents[i].location[0,1] < -130 and a.afferents[i].location[0,1] > -170:
            hindfoot.append(a.afferents[i])
    
    forefoot_population = fs.classes.AfferentPopulation()
    hindfoot_population = fs.classes.AfferentPopulation()

    hindfoot_population.afferents = hindfoot
    forefoot_population.afferents = forefoot

    return forefoot_population, hindfoot_population

def angle_polynomial():

    angles = np.array([-5, 0, 5])

    # Forefoot and hindfoot pressures
    forefoot_pressures = np.array([8.1, 10.0, 12.2])
    hindfoot_pressures = np.array([12.4, 10.9, 8.9])

    # Generate scale factor to match with simulation
    model_mass = 74
    study_avg_mass = 64.5
    scale_factor = model_mass / study_avg_mass

    scaled_forefoot_pressures = forefoot_pressures * scale_factor
    scaled_hindfoot_pressures = hindfoot_pressures * scale_factor

    # Fit 2nd-degree polynomials
    forefoot_poly = np.poly1d(np.polyfit(angles, scaled_forefoot_pressures, 2))
    hindfoot_poly = np.poly1d(np.polyfit(angles, scaled_hindfoot_pressures, 2))

    return forefoot_poly, hindfoot_poly

def pressure_to_delta_length(pressure_forefoot, pressure_hindfoot):

        # pressure is in N/cm^2
        pressure_forefoot /= 100
        pressure_hindfoot /= 100
        length_forefoot = 10 #in mm
        length_hindfoot = 30 #in mm
        youngs_forefoot = 0.29 # N/mm^2
        youngs_hindfoot = 0.33 # N/mm^2

        delta_length_forefoot = (pressure_forefoot*length_forefoot)/(youngs_forefoot)
        delta_length_hindfoot = (pressure_hindfoot*length_hindfoot)/(youngs_hindfoot)

        return delta_length_forefoot, delta_length_hindfoot

def get_firing_rate(angle, forefoot_poly, hindfoot_poly, forefoot_population, hindfoot_population, forefoot_len, hindfoot_len):

    pressure_forefoot = forefoot_poly(angle)
    pressure_hindfoot = hindfoot_poly(angle)
    
    delta_length_forefoot, delta_length_hindfoot = pressure_to_delta_length(pressure_forefoot, pressure_hindfoot)

    # delta_length_forefoot_zero, delta_length_hindfoot_zero = pressure_to_delta_length(forefoot_poly(0), hindfoot_poly(0))

    forefoot_stimulus = fs.stim_linear(len_prev=forefoot_len, amp=delta_length_forefoot+10-forefoot_len, fs=100000, loc=(0,-10))
    hindfoot_stimulus = fs.stim_linear(len_prev=hindfoot_len, amp=delta_length_hindfoot+30-hindfoot_len, fs=100000, loc=(0,-150))
    forefoot_len =  delta_length_forefoot+10
    hindfoot_len =  delta_length_hindfoot+30

    response = forefoot_population.response(forefoot_stimulus)
    fr_forefoot = np.mean(response.rate())
    response = hindfoot_population.response(hindfoot_stimulus)
    fr_hindfoot = np.mean(response.rate())

    return fr_forefoot, fr_hindfoot, forefoot_len, hindfoot_len
