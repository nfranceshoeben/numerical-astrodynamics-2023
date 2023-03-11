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
# Simulation 1 ############################################################
###########################################################################

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Load spice kernels.
spice.load_standard_kernels()
spice.load_kernel( current_directory + "/Assignment1/juice_mat_crema_5_1_150lb_v01.bsp" );

# Create settings for celestial bodies
bodies_to_create = ['Ganymede', 'Jupiter']
global_frame_origin = 'Jupiter'
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
central_bodies = ['Jupiter']

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
    observer_body_name='Jupiter',
    reference_frame_name='ECLIPJ2000',
    aberration_corrections='NONE',
    ephemeris_time = simulation_start_epoch )

# Define required outputs
dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede'),
                               propagation_setup.dependent_variable.relative_position('Ganymede', 'Jupiter')]

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
         filename='JUICEPropagationHistory_Q4_1.dat',
         directory='./'
         )

save2txt(solution=dependent_variables,
         filename='JUICEPropagationHistory_DependentVariables_Q4_1.dat',
         directory='./'
         )

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# Extract time and Kepler elements from dependent variables
kepler_elements_1 = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]

###########################################################################
# Simulation 2 ############################################################
###########################################################################

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Load spice kernels.
spice.load_standard_kernels()
spice.load_kernel( current_directory + "/Assignment1/juice_mat_crema_5_1_150lb_v01.bsp" );

# Create settings for celestial bodies
bodies_to_create = ['Ganymede', 'Jupiter', 'Sun','Saturn','Europa','Io','Callisto']
global_frame_origin = 'Jupiter'
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
central_bodies = ['Jupiter']

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
    observer_body_name='Jupiter',
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
                               propagation_setup.dependent_variable.spherical_harmonic_terms_acceleration_norm('JUICE','Jupiter',jupiter_spherical_accelerations),
                               propagation_setup.dependent_variable.relative_position('Ganymede', 'Jupiter')
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
         filename='JUICEPropagationHistory_Q4_2.dat',
         directory='./'
         )

save2txt(solution=dependent_variables,
         filename='JUICEPropagationHistory_DependentVariables_Q4_2.dat',
         directory='./'
         )

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# Extract time and Kepler elements from dependent variables
kepler_elements_2 = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]

# Data from q1 and q2:

position_elements_q1 = np.genfromtxt('JUICEPropagationHistory_Q1.dat', usecols=[1,2,3])
position_elements_q2 = np.genfromtxt('JUICEPropagationHistory_Q2.dat',usecols=[1,2,3])

ganymede_position_elements_1 = kepler_elements_1[:,-3:]
ganymede_position_elements_2 = kepler_elements_2[:,-3:]

r_J_sc_1 = np.genfromtxt('JUICEPropagationHistory_Q4_1.dat',usecols=[1,2,3])
r_J_sc_2 = np.genfromtxt('JUICEPropagationHistory_Q4_2.dat',usecols=[1,2,3])

r_G_sc_1 = r_J_sc_1 - ganymede_position_elements_1
r_G_sc_2 = r_J_sc_2 - ganymede_position_elements_2

diff1 = position_elements_q1 - r_G_sc_1
diff2 = position_elements_q2 - r_G_sc_2

# Plotting:

diff1Total = np.linalg.norm(diff1,axis=1)
diff2Total = np.linalg.norm(diff2,axis=1)

figComparison = plt.figure()
ax1 = figComparison.add_subplot(2,1,1)
ax1.set_title('Difference between results i and ii')
ax1.set_xlabel('Time [days]')
ax1.set_ylabel('Difference [m]')
ax1.set_xlim([min(time_days),max(time_days)])
ax1.grid()
ax1.plot(time_days, diff1Total)
ax2 = figComparison.add_subplot(2,1,2)
ax2.set_title('Difference between results iii and iv')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('Difference [m]')
ax2.set_xlim([min(time_days),max(time_days)])
ax2.grid()
ax2.plot(time_days, diff2Total)

plt.show()

