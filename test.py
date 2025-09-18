import footsim as fs
import numpy as np
from footsim.plotting import plot
from firing_rates import afferent_populations, angle_polynomial, pressure_to_delta_length, get_firing_rate
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import holoviews as hv
hv.extension('mpl')

def do_it(angles):

    forefoot_population, hindfoot_population = afferent_populations()

    forefoot_poly, hindfoot_poly = angle_polynomial()

    initial_forefoot_pressure = forefoot_poly(0)
    initial_hindfoot_pressure = hindfoot_poly(0) 

    delta_length_forefoot, delta_length_hindfoot = pressure_to_delta_length(initial_forefoot_pressure, initial_hindfoot_pressure)

    forefoot_len = 10+delta_length_forefoot
    hindfoot_len = 30+delta_length_hindfoot

    afr_forefoot = []
    afr_hindfoot = []

    for i in range(len(angles)):
        
        fr_forefoot, fr_hindfoot, forefoot_len, hindfoot_len = get_firing_rate(angles[i], forefoot_poly, hindfoot_poly, forefoot_population, hindfoot_population, forefoot_len, hindfoot_len)
        afr_forefoot.append(fr_forefoot)
        afr_hindfoot.append(fr_hindfoot)
    
    return afr_forefoot, afr_hindfoot

if __name__ == "__main__":

    # angles = []
    steps = 200
    angle = []

    curr_angle = 0
    angle.append(curr_angle)
    for i in range(int(steps/4)):
        curr_angle += 20/steps
        angle.append(curr_angle)

    for i in range(int(steps/2)):
        curr_angle -= 20/steps
        angle.append(curr_angle)

    for i in range(int(steps/4)):
        curr_angle += 20/steps
        angle.append(curr_angle)

    afr_forefoot, afr_hindfoot = do_it(angle)
    
    # print(afr_forefoot)
    plt.plot(afr_forefoot)
    plt.show()
    plt.plot(afr_hindfoot)
    plt.show()