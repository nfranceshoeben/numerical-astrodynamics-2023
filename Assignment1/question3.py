###########################################################################
#
# # Numerical Astrodynamics 2022/2023
#
# # Assignment 1 - Propagation Settings
#
###########################################################################


''' 
Copyright (c) 2010-2020, Delft University of Technology
All rights reserved

This file is part of Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

import os

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup


####################################################################################
####################################################################################

'''
        _______ _______ ______ _   _ _______ _____ ____  _   _ 
     /\|__   __|__   __|  ____| \ | |__   __|_   _/ __ \| \ | |
    /  \  | |     | |  | |__  |  \| |  | |    | || |  | |  \| |
   / /\ \ | |     | |  |  __| | . ` |  | |    | || |  | | . ` |
  / ____ \| |     | |  | |____| |\  |  | |   _| || |__| | |\  |
 /_/   _\_\_|__ __|_|  |______|_| \_|  |_|  |_____\____/|_| \_|
 | |  | |  ____|  __ \|  ____|                                 
 | |__| | |__  | |__) | |__                                    
 |  __  |  __| |  _  /|  __|                                   
 | |  | | |____| | \ \| |____                                  
 |_|__|_|______|_|__\_\______|   _____ ______                  
 |  __ \| |    |  ____|   /\    / ____|  ____|                 
 | |__) | |    | |__     /  \  | (___ | |__                    
 |  ___/| |    |  __|   / /\ \  \___ \|  __|                   
 | |    | |____| |____ / ____ \ ____) | |____                  
 |_|    |______|______/_/    \_\_____/|______|                 
                                                               
                                                            
'''


# The Big Switch of Question 3:

Question3Switch = True #If false, follows computation 1 of question 3. If true, follows 2.

# Retrieve current directory
current_directory = os.getcwd()

# # student number: 1244779 --> 1244ABC
A = 5
B = 5
C = 4

simulation_start_epoch = 35.4 * constants.JULIAN_YEAR + A * 7.0 * constants.JULIAN_DAY + B * constants.JULIAN_DAY + C * constants.JULIAN_DAY / 24.0
simulation_end_epoch = simulation_start_epoch + 344.0 * constants.JULIAN_DAY / 24.0

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Load spice kernels.
spice.load_standard_kernels()
spice.load_kernel( current_directory + "/Assignment1/juice_mat_crema_5_1_150lb_v01.bsp" );

# Create settings for celestial bodies
bodies_to_create = ['Ganymede', 'Jupiter']
global_frame_origin = 'Ganymede'
global_frame_orientation = 'ECLIPJ2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Atmosphere
density_scale_height = 40e3
#constant_temperature = 290
density_at_zero_altitude = 2e-9
# create atmosphere settings and add to body settings of "Earth"
body_settings.get( "Ganymede" ).atmosphere_settings = environment_setup.atmosphere.exponential(
     density_scale_height, density_at_zero_altitude)

# Create environment
bodies = environment_setup.create_system_of_bodies(body_settings)

###########################################################################
# CREATE VEHICLE ##########################################################
###########################################################################

# Create vehicle object
bodies.create_empty_body( 'JUICE' )
bodies.get("JUICE").mass = 2000.0

reference_area = 100.0
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area, [drag_coefficient, 0, 0]
)
environment_setup.add_aerodynamic_coefficient_interface(
    bodies, "JUICE", aero_coefficient_settings)



###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies = ['Ganymede']

# Define accelerations acting on vehicle.

if Question3Switch == False:
    acceleration_settings_on_vehicle = dict(
        Ganymede=[propagation_setup.acceleration.point_mass_gravity()],
        Jupiter=[propagation_setup.acceleration.spherical_harmonic_gravity(4,0)]        
    )

elif Question3Switch:

    acceleration_settings_on_vehicle = dict(
        Ganymede=[propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.aerodynamic()]
    )



# Create global accelerations dictionary.
acceleration_settings = {'JUICE': acceleration_settings_on_vehicle}

# Create acceleration models.
acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

###########################################################################
# CREATE PROPAGATION SETTINGS #############################################
###########################################################################

# Define initial state.
system_initial_state = spice.get_body_cartesian_state_at_epoch(
    target_body_name='JUICE',
    observer_body_name='Ganymede',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time = simulation_start_epoch )

ganymede_spherical_accelerations = [ (0,0), (2,0), (2,2) ]
jupiter_spherical_accelerations = [ (0,0), (2,0), (4,0) ]


# Define required outputs

if Question3Switch == False:
    dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede'), 
                                propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type,'JUICE','Ganymede'),
                                propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.spherical_harmonic_gravity_type,'JUICE','Jupiter')]
                                

elif Question3Switch:
        dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede'), 
                                propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type,'JUICE','Ganymede'),
                                propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.aerodynamic_type,'JUICE','Ganymede')]


#Saving: Kepler elements (a,e,i,arg,raan,theta). If Question3Switch==False: Ganymede point mass acceleration, Jupiter spherical harmonic acceleration (4,0).
#If Question3Switch==True: Ganymede Point Mass acceleration, Ganymede aerodynamic



# Create numerical integrator settings.
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    fixed_step_size
)

# Create propagation settings.
termination_settings = propagation_setup.propagator.time_termination( simulation_end_epoch )
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    system_initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    output_variables = dependent_variables_to_save
)

propagator_settings.print_settings.print_initial_and_final_conditions = True


###########################################################################
# PROPAGATE ORBIT #########################################################
###########################################################################

# Create simulation object and propagate dynamics.
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings )

# Retrieve all data produced by simulation
propagation_results = dynamics_simulator.propagation_results

# Extract numerical solution for states and dependent variables
state_history = propagation_results.state_history
dependent_variables = propagation_results.dependent_variable_history

###########################################################################
# SAVE RESULTS ############################################################
###########################################################################


if Question3Switch==False:
    save2txt(solution=state_history,
            filename='JUICEPropagationHistory_Q3_1.dat',
            directory='./'
            )

    save2txt(solution=dependent_variables,
            filename='JUICEPropagationHistory_DependentVariables_Q3_1.dat',
            directory='./'
            )
    
if Question3Switch==True:
    save2txt(solution=state_history,
            filename='JUICEPropagationHistory_Q3_2.dat',
            directory='./'
            )

    save2txt(solution=dependent_variables,
            filename='JUICEPropagationHistory_DependentVariables_Q3_2.dat',
            directory='./'
            )

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# Extract time and Kepler elements from dependent variables
kepler_elements = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]

#Saving: Kepler elements (a,e,i,arg,raan,theta). If Question3Switch==False: Ganymede point mass acceleration, Jupiter spherical harmonic acceleration (4,0).
#If Question3Switch==True: Ganymede Point Mass acceleration, Ganymede aerodynamic

unpert = np.genfromtxt('JUICEPropagationHistory_Q1.dat')
r1 = np.genfromtxt('JUICEPropagationHistory_Q3_1.dat')
r2 = np.genfromtxt('JUICEPropagationHistory_Q3_2.dat')

a1 = np.genfromtxt('JUICEPropagationHistory_DependentVariables_Q3_1.dat')[:,6:]
a2 = np.genfromtxt('JUICEPropagationHistory_DependentVariables_Q3_2.dat')[:,6:]

aAbs1 = np.linalg.norm(a1,axis=1)
aAbs2 = np.linalg.norm(a2,axis=1)

a1Cum = sp.integrate.cumulative_trapezoid(aAbs1,dx=10.0)
a2Cum = sp.integrate.cumulative_trapezoid(aAbs2,dx=10.0)

dr1 = r1[:,1:-3] - unpert[:,1:-3]
dr2 = r2[:,1:-3] - unpert[:,1:-3]

drAbs1 = np.linalg.norm(dr1,axis=1)
drAbs2 = np.linalg.norm(dr2,axis=1)

epsilon1 = drAbs1[1:]/a1Cum
epsilon2 = drAbs2[1:]/a2Cum

#Plotting

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(time_days[1:],epsilon1, label='Effectiveness 1')
ax1.plot(time_days[1:],epsilon2, label='Effectiveness 2')
ax1.grid()
ax1.legend()
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Effectiveness')
ax1.set_xlim(min(time_days[1:]),max(time_days))

plt.show()






