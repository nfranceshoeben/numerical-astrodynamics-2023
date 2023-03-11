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
from matplotlib import pyplot as plt

from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup

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
bodies_to_create = ['Ganymede']
global_frame_origin = 'Ganymede'
global_frame_orientation = 'ECLIPJ2000'
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create environment
bodies = environment_setup.create_system_of_bodies(body_settings)

###########################################################################
# CREATE VEHICLE ##########################################################
###########################################################################

# Create vehicle object
bodies.create_empty_body( 'JUICE' )


###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies = ['Ganymede']

# Define accelerations acting on vehicle.
acceleration_settings_on_vehicle = dict(
    Ganymede=[propagation_setup.acceleration.point_mass_gravity()]
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

# Define required outputs
dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede')]

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

save2txt(solution=state_history,
         filename='JUICEPropagationHistory_Q1.dat',
         directory='./'
         )

save2txt(solution=dependent_variables,
         filename='JUICEPropagationHistory_DependentVariables_Q1.dat',
         directory='./'
         )

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# Extract time and Kepler elements from dependent variables
kepler_elements = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]

kepler_elements_difference = np.zeros([len(kepler_elements[:,0]),len(kepler_elements[0,:])])

for i in range(len(kepler_elements[:,0])):
    kepler_elements_difference[i,:] = kepler_elements[i,:] - kepler_elements[1,:]


# Plotting:

'''
fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
ax.plot(time_days,kepler_elements_difference[:,0])
ax.set_xlabel('Time [days]')
ax.set_ylabel('Semi-major axis [m]')
ax.grid()
ax.set_xlim(min(time_days),max(time_days))

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Eccentricity')
ax2.plot(time_days,kepler_elements_difference[:,1])
ax2.grid()
ax2.set_xlim(min(time_days),max(time_days))

fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(time_days,kepler_elements_difference[:,2])
ax3.set_xlabel('Time [days]')
ax3.set_ylabel('Inclination [rad]')
ax3.grid()
ax3.set_xlim(min(time_days),max(time_days))

fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(time_days,kepler_elements_difference[:,3])
ax4.set_xlabel('Time [days]')
ax4.set_ylabel('Argument of periapsis [rad]')
ax4.grid()
ax4.set_xlim(min(time_days),max(time_days))

fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)
ax5.plot(time_days,kepler_elements_difference[:,4], c='C01')
ax5.set_xlabel('Time [days]')
ax5.set_ylabel('Right Ascension of the Ascending Node [rad]')
ax5.grid()
ax5.set_xlim(min(time_days),max(time_days))

fig6 = plt.figure()
ax6 = fig6.add_subplot(1,1,1)
ax6.plot(time_days,kepler_elements_difference[:,5])
ax6.set_xlabel('Time [days]')
ax6.set_ylabel('True Anomaly [rad]')
ax6.grid()
ax6.set_xlim(min(time_days),max(time_days))
'''

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = ax1.twinx()
ax2.plot(time_days,kepler_elements_difference[:,1], 'C1', label='Eccentricity axis difference')
ax1.plot(time_days,kepler_elements_difference[:,0], 'C0',label='Semi-major axis difference')
#ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Semi-major axis difference [m]')
ax2.set_ylabel('Eccentricity difference')
ax1.grid()
ax1.set_xlim(min(time_days),max(time_days))
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=1)

ax3 = fig.add_subplot(2,1,2)
ax4 = ax3.twinx()
ax3.plot(time_days,kepler_elements_difference[:,2], 'C1', label='Inclination difference')
ax3.plot(time_days,kepler_elements_difference[:,3], 'C0', label='Argument of periapsis difference')
ax3.plot(time_days,kepler_elements_difference[:,4], 'C2', label='RAAN difference')
ax4.plot(time_days,kepler_elements_difference[:,5], 'C3--', label='True Anomaly difference')
ax3.grid()
ax3.set_xlabel('Time [days]')
ax3.set_ylabel('Difference [rad]')
ax3.set_xlim(min(time_days),max(time_days))
ax3.grid()
ax4.set_ylabel('Difference [rad]')
ax4.yaxis.label.set_color('C1')
lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
leg = ax4.legend(lines + lines2, labels + labels2, loc=1)
i = 0
for text in leg.get_texts():

    if i==3:
        text.set_color("C1")
    i +=1


plt.show()







