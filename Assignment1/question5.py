###########################################################################
#
# # Numerical Astrodynamics 2022/2023
#
# # Assignment 1, Question 5 - Propagation Settings
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
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion


class ThrustGuidance:

    def __init__(self,
                 maximum_thrust: float, # Maximum thrust value that is to be used
                 true_anomaly_threshold: float, # Limiting value of true anomaly before and after node at which thrust should be on/off
                 bodies: environment.SystemOfBodies):
        self.maximum_thrust = maximum_thrust
        self.true_anomaly_threshold = true_anomaly_threshold
        self.bodies = bodies

    def compute_thrust_direction(self, current_time: float):

        # Check if computation is to be done. NOTE TO STUDENTS: ALL CALCULATION OF THRUST DIRECTION MUST BE INSIDE
        # THE FOLLOWING BLOCK
        if( current_time == current_time ):

            # Retrieve current JUICE Cartesian state w.r.t. Ganymede from environment
            current_cartesian_state = self.bodies.get_body( 'JUICE' ).state - self.bodies.get_body( 'Ganymede' ).state
            gravitational_parameter = self.bodies.get_body( 'Ganymede' ).gravitational_parameter
            current_keplerian_state = element_conversion.cartesian_to_keplerian( current_cartesian_state, gravitational_parameter )

            # Compute and return current thrust direction (3x1 vector)
            radial_unit_vector = current_cartesian_state[0:3]/np.linalg.norm(current_cartesian_state[0:3])
            vel_unit_vector = current_cartesian_state[3:]/np.linalg.norm(current_cartesian_state[3:])

            if (abs(current_keplerian_state[5]-(6.28318-current_keplerian_state[3]))<self.true_anomaly_threshold):
                thrust_direction = np.cross(radial_unit_vector,vel_unit_vector)
            #XXXX
            elif (abs(current_keplerian_state[5]+current_keplerian_state[3]-3.141592)<self.true_anomaly_threshold):
                thrust_direction = -np.cross(radial_unit_vector,vel_unit_vector)

            else:
                thrust_direction = np.array([1,0,0])

            # Here, the direction of the thrust (in a frame with inertial orientation; same as current_cartesian_state)
            # should be returned as a numpy unit vector (3x1)
            return thrust_direction

        # If no computation is to be done, return zeros
        else:
            return np.zeros([3,1])
    def compute_thrust_magnitude(self, current_time: float):

        # Check if computation is to be done. NOTE TO STUDENTS: ALL CALCULATION OF THRUST MAGNITUDE MUST BE INSIDE
        # THE FOLLOWING BLOCK
        if( current_time == current_time ):

            # Retrieve current JUICE Cartesian  and Keplerian state w.r.t. Ganymede from environment
            current_cartesian_state = self.bodies.get_body( 'JUICE' ).state - self.bodies.get_body( 'Ganymede' ).state
            gravitational_parameter = self.bodies.get_body( 'Ganymede' ).gravitational_parameter
            current_keplerian_state = element_conversion.cartesian_to_keplerian( current_cartesian_state, gravitational_parameter )

            # Compute and return current thrust magnitude (scalar)
            #test1 = abs(current_keplerian_state[5]-(6.28318-current_keplerian_state[3]))
            #test2 = abs(current_keplerian_state[5]+current_keplerian_state[3]-3.141592)

            #if (abs(current_keplerian_state[5]-current_keplerian_state[4]) < self.true_anomaly_threshold and abs(current_keplerian_state[5]-current_keplerian_state[4]) > 0) or (abs(current_keplerian_state[5]-current_keplerian_state[4]) > 6.24828 and abs(current_keplerian_state[5]-current_keplerian_state[4]) < 0) or (abs(current_keplerian_state[5]-current_keplerian_state[4]) > 3.10669 and abs(current_keplerian_state[5]-current_keplerian_state[4]) < 3.1765):
            if (abs(current_keplerian_state[5]-(6.28318-current_keplerian_state[3]))<self.true_anomaly_threshold) or (abs(current_keplerian_state[5]+current_keplerian_state[3]-3.141592)<self.true_anomaly_threshold):
                thrust_magnitude = 1.0
            #XXXX

            else:
                thrust_magnitude = 0.0

            # Here, the value of the thrust magnitude (in Newtons, as a single floating point variable), should be returned
            return thrust_magnitude
        # If no computation is to be done, return zeros
        else:
            return 0.0




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
spice.load_kernel( current_directory + "\Assignment1\juice_mat_crema_5_1_150lb_v01.bsp" );

# Create settings for celestial bodies
bodies_to_create = ['Ganymede','Jupiter','Sun','Saturn','Europa','Io','Callisto']
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
# CREATE THRUST MODEL #####################################################
###########################################################################

# Create thrust guidance object (e.g. object that calculates direction/magnitude of thrust)
thrust_magnitude = 1.0
true_anomaly_threshold = 0.034907 #2deg in radians
thrust_guidance_object = ThrustGuidance( thrust_magnitude, true_anomaly_threshold, bodies )

# Create engine model (default JUICE-fixed pointing direction) with custom thrust magnitude calculation
constant_specific_impulse = 300
thrust_magnitude_settings = (
    propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(
        thrust_guidance_object.compute_thrust_magnitude,
        constant_specific_impulse ) )
environment_setup.add_engine_model(
    'JUICE', 'MainEngine', thrust_magnitude_settings, bodies )

# Create vehicle rotation model such that thrust points in required direction in inertial frame
thrust_direction_function = thrust_guidance_object.compute_thrust_direction
rotation_model_settings = environment_setup.rotation_model.custom_inertial_direction_based(
    thrust_direction_function,
    "JUICE-fixed",
    "ECLIPJ2000" )
environment_setup.add_rotation_model( bodies, "JUICE", rotation_model_settings)


###########################################################################
# CREATE ACCELERATIONS ####################################################
###########################################################################

# Define bodies that are propagated, and their central bodies of propagation.
bodies_to_propagate = ['JUICE']
central_bodies = ['Ganymede']

# Define accelerations acting on vehicle.
acceleration_settings_on_vehicle = dict(
    JUICE=[propagation_setup.acceleration.thrust_from_engine('MainEngine')],
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

# Define required outputs
dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('JUICE','Ganymede'),
                               propagation_setup.dependent_variable.body_mass('JUICE'),
                               propagation_setup.dependent_variable.total_mass_rate('JUICE'),
                               propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.thrust_acceleration_type, 'JUICE','JUICE'),
                               propagation_setup.dependent_variable.single_acceleration_norm(propagation_setup.acceleration.thrust_acceleration_type, 'JUICE', 'JUICE')]

# Create numerical integrator settings.
fixed_step_size = 10.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    fixed_step_size
)

# Create translational propagation settings.
termination_settings = propagation_setup.propagator.time_termination( simulation_end_epoch )
translational_propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    system_initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_settings,
    output_variables = dependent_variables_to_save
)

# Create a mass rate model so that the vehicle loses mass according to how much thrust acts on it
mass_rate_settings = dict(JUICE=[propagation_setup.mass_rate.from_thrust()])
mass_rate_models = propagation_setup.create_mass_rate_models(
    bodies,
    mass_rate_settings,
    acceleration_models
)

# Create mass propagator settings
mass_propagator_settings = propagation_setup.propagator.mass(
    bodies_to_propagate,
    mass_rate_models,
    [2e3], # initial vehicle mass
    simulation_start_epoch,
    integrator_settings,
    termination_settings )

# Create combined mass and translational dynamics propagator settings
propagator_settings = propagation_setup.propagator.multitype(
    [translational_propagator_settings, mass_propagator_settings],
    integrator_settings,
    simulation_start_epoch,
    termination_settings,
    output_variables = dependent_variables_to_save)


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
         filename='JUICEPropagationHistory_Q5.dat',
         directory='./'
         )

save2txt(solution=dependent_variables,
         filename='JUICEPropagationHistory_DependentVariables_Q5.dat',
         directory='./'
         )

###########################################################################
# PLOT RESULTS ############################################################
###########################################################################

# Extract time and Kepler elements from dependent variables
kepler_elements = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - simulation_start_epoch / constants.JULIAN_DAY for t in time ]
time_hours = np.dot(time_days,24)

kepler_elements_q2 = np.genfromtxt('JUICEPropagationHistory_DependentVariables_Q2.dat',usecols=(1,2,3,4,5,6))
kepler_elements_q5 = kepler_elements[:,0:6]
diff = kepler_elements_q5-kepler_elements_q2

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(time_hours,kepler_elements[:,-1])
ax1.set_xlim(time_hours[0],time_hours[-1])
ax1.set_xlabel('Time [hours]')
ax1.set_ylabel('Acceleration [m/s^2]')
ax1.grid()

fig2 = plt.figure()
ax1 = fig2.add_subplot(1,1,1)
ax1.plot(time_hours,kepler_elements[:,6]) #Make it -6 if it doesn't work tomorrow morning
ax1.set_xlim(time_hours[0],time_hours[-1])
ax1.set_xlabel('Time [hours]')
ax1.set_ylabel('Vehicle mass [kg]')
ax1.grid()

fig3 = plt.figure()
ax1 = fig3.add_subplot(2,3,1)
ax2 = fig3.add_subplot(2,3,2)
ax3 = fig3.add_subplot(2,3,3)
ax4 = fig3.add_subplot(2,3,4)
ax5 = fig3.add_subplot(2,3,5)
ax6 = fig3.add_subplot(2,3,6)

ax1.plot(time_hours,diff[:,0])
ax2.plot(time_hours,diff[:,1])
ax3.plot(time_hours,diff[:,2])
ax4.plot(time_hours,diff[:,3])
ax5.plot(time_hours,diff[:,4])
ax6.plot(time_hours,diff[:,5])

ax1.set_xlabel('Time [hours]')
ax2.set_xlabel('Time [hours]')
ax3.set_xlabel('Time [hours]')
ax4.set_xlabel('Time [hours]')
ax5.set_xlabel('Time [hours]')
ax6.set_xlabel('Time [hours]')

ax1.set_ylabel('Semi-major axis [m]')
ax2.set_ylabel('Eccentricity')
ax3.set_ylabel('Inclination [rad]')
ax4.set_ylabel('Argument of Periapsis [rad]')
ax5.set_ylabel('RAAN [rad]')
ax6.set_ylabel('True anomaly [rad]')

ax1.set_title('Semi-major axis difference')
ax2.set_title('Eccentricity axis difference')
ax3.set_title('Inclination axis difference')
ax4.set_title('Argument of Periapsis axis difference')
ax5.set_title('RAAN difference')
ax6.set_title('True anomaly difference')

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()

fig4 = plt.figure()
ax1 = fig4.add_subplot(1,1,1)
ax1.plot(time_hours[0:2200],kepler_elements_q5[:2200,3])
ax1.set_xlabel('Time [hours]')
ax1.set_ylabel('Inclination [rad]')
ax1.grid()
ax1.set_xlim(time_hours[0],time_hours[2200])


plt.show()

print(kepler_elements_q5[500:512,3])