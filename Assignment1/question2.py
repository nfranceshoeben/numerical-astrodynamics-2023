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
bodies_to_create = ['Ganymede', 'Jupiter', 'Sun','Saturn','Europa','Io','Callisto']
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

# Create radiation pressure settings, and add to vehicle
reference_area_radiation = 100.0
radiation_pressure_coefficient = 1.2
occulting_bodies = ["Ganymede"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)
environment_setup.add_radiation_pressure_interface(
    bodies, "JUICE", radiation_pressure_settings)


###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies = ['Ganymede']

# Define accelerations acting on vehicle.
acceleration_settings_on_vehicle = dict(
    Ganymede=[propagation_setup.acceleration.spherical_harmonic_gravity(2,2),
              propagation_setup.acceleration.aerodynamic()],
    Jupiter=[propagation_setup.acceleration.spherical_harmonic_gravity(4,0)],
    Sun=[propagation_setup.acceleration.point_mass_gravity(),
         propagation_setup.acceleration.cannonball_radiation_pressure()],
    Saturn=[propagation_setup.acceleration.point_mass_gravity()],
    Europa=[propagation_setup.acceleration.point_mass_gravity()],
    Io=[propagation_setup.acceleration.point_mass_gravity()],
    Callisto=[propagation_setup.acceleration.point_mass_gravity()]
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
dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede'), 
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.spherical_harmonic_gravity_type,'JUICE','Ganymede'),
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.aerodynamic_type,'JUICE','Ganymede'),
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.spherical_harmonic_gravity_type,'JUICE','Jupiter'),
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type,'JUICE','Sun'),
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.cannonball_radiation_pressure_type,'JUICE','Sun'),
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type,'JUICE','Saturn'),
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type,'JUICE','Europa'),
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type,'JUICE','Io'),
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.point_mass_gravity_type,'JUICE','Callisto'),
                               propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm('JUICE','Ganymede',ganymede_spherical_accelerations),
                               propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm('JUICE','Jupiter',jupiter_spherical_accelerations)
                               ]


#Saving: Kepler elements(a,e,i,arg,RAAN,theta),norms:Ganymede gravity, Ganymede atmosphere, Jupiter Gravity, Sun Gravity, Sun Pressure, 
# Saturn Gravity, Europa Gravity, Io Gravity, Callisto Gravity, Ganymede Spherical 0,0, Ganymede Spherical 2,0, Ganymede Spherical 2,2,
# Jupiter Spherical 0,0, Jupiter Spherical 2,0, Jupiter Spherical 4,0



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
         filename='JUICEPropagationHistory_Q2.dat',
         directory='./'
         )

save2txt(solution=dependent_variables,
         filename='JUICEPropagationHistory_DependentVariables_Q2.dat',
         directory='./'
         )

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# Extract time and Kepler elements from dependent variables
kepler_elements = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]


# Plotting:

fig1 = plt.figure()
ax = fig1.add_subplot(1,1,1)
ax.plot(time_days,kepler_elements[:,0])
ax.set_xlabel('Time [days]')
ax.set_ylabel('Semi-major axis [m]')

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Eccentricity')
ax2.plot(time_days,kepler_elements[:,1])

fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(time_days,kepler_elements[:,2])
ax3.set_xlabel('Time [days]')
ax3.set_ylabel('Inclination [rad]')

fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(time_days,kepler_elements[:,3])
ax4.set_xlabel('Time [days]')
ax4.set_ylabel('Argument of periapsis [rad]')

fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)
ax5.plot(time_days,kepler_elements[:,4], c='C01')
ax5.set_xlabel('Time [days]')
ax5.set_ylabel('Right Ascension of the Ascending Node [rad]')

fig6 = plt.figure()
ax6 = fig6.add_subplot(1,1,1)
ax6.plot(time_days,kepler_elements[:,5])
ax6.set_xlabel('Time [days]')
ax6.set_ylabel('True Anomaly [rad]')

#Saving: Kepler elements(a,e,i,arg,RAAN,theta),norms:Ganymede gravity, Ganymede atmosphere, Jupiter Gravity, Sun Gravity, Sun Pressure, 
# Saturn Gravity, Europa Gravity, Io Gravity, Callisto Gravity, Ganymede Spherical 0,0, Ganymede Spherical 2,0, Ganymede Spherical 2,2,
# Jupiter Spherical 0,0, Jupiter Spherical 2,0, Jupiter Spherical 4,0

fig7 = plt.figure()
ax1 = fig7.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(time_days,kepler_elements[:,6], 'C0', label='Ganymede Gravitational Acceleration')
ax2.plot(time_days,kepler_elements[:,7], 'C0--', label='Ganymede Aerodynamic Acceleration')
ax2.plot(time_days,kepler_elements[:,8], 'C1',label='Jupiter Gravitational Acceleration')
ax2.plot(time_days,kepler_elements[:,9], 'C2',label='Sun Gravitational Acceleration')
ax2.plot(time_days,kepler_elements[:,10], 'C2--', label='Sun Radiation Pressure')
ax2.plot(time_days,kepler_elements[:,11], 'C3',label='Saturn Gravitational Acceleration')
ax2.plot(time_days,kepler_elements[:,12], 'C4',label='Europa Gravitational Acceleration')
ax2.plot(time_days,kepler_elements[:,13], 'C5',label='Io Gravitational Acceleration')
ax2.plot(time_days,kepler_elements[:,14], 'C6',label='Callisto Gravitational Acceleration')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
test = lines+lines2
test2 = labels+labels2
leg = ax2.legend(test, test2, loc=1)
i = 0
for text in leg.get_texts():

    if i!=0:
        text.set_color("C1")
    i +=1

ax1.grid()
ax1.set_xlim([min(time_days),max(time_days)])
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Acceleration [m/s^2]')
ax2.set_ylabel('Acceleration [m/s^2]')
ax2.yaxis.label.set_color('C1')
#ax1.legend()

fig8 = plt.figure()
ax1 = fig8.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(time_days,kepler_elements[:,15],label='Ganymede (0,0)')
ax2.plot(time_days,kepler_elements[:,16], 'C1', label='Ganymede (2,0)')
ax2.plot(time_days,kepler_elements[:,17], 'C2', label='Ganymede (2,2)')
ax2.plot(time_days,kepler_elements[:,18], 'C3', label='Jupiter (0,0)')
ax2.plot(time_days,kepler_elements[:,19], 'C4', label='Jupiter (2,0)')
ax2.plot(time_days,kepler_elements[:,20], 'C5', label='Jupiter (4,0)')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Acceleration [m/s^2]')
ax2.set_ylabel('Acceleration [m/s^2]')
ax2.yaxis.label.set_color('C1')
ax1.grid()
ax1.set_xlim(min(time_days),max(time_days))
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
test = lines+lines2
test2 = labels+labels2
leg = ax2.legend(test, test2, loc=1)
i = 0
for text in leg.get_texts():

    if i!=0:
        text.set_color("C1")
    i +=1

time_hours = np.multiply(time_days,24)

fig9 = plt.figure()
ax1 = fig9.add_subplot(2,1,1)
ax1.plot(time_hours,kepler_elements[:,9],label='Sun Gravitational Acceleration')
ax1.set_xlabel('Time [hours]')
ax1.set_ylabel('Acceleration [m/s^2]')
ax1.set_xlim(min(time_hours), max(time_hours))
ax1.grid()

ax2 = fig9.add_subplot(2,1,2)
ax2.plot(time_hours,kepler_elements[:,13],label='Io Gravitational Acceleration')
ax2.set_xlabel('Time [hours]')
ax2.set_ylabel('Acceleration [m/s^2]')
ax2.set_xlim(min(time_hours), max(time_hours))
ax2.grid()

plt.show()








