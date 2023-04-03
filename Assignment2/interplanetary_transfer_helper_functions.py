''' 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''


import numpy as np
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import estimation_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.astro import element_conversion
from tudatpy.util import result2array

# Define departure/arrival epoch - in seconds since J2000
departure_epoch = 2055.746011*86400
time_of_flight = 215.3577652*86400
arrival_epoch = departure_epoch + time_of_flight
target_body = 'Mars'
global_frame_orientation = 'ECLIPJ2000'
fixed_step_size = 3600.0

################ HELPER FUNCTIONS: DO NOT MODIFY ########################################

# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def write_propagation_results_to_file(
        dynamics_simulator: numerical_simulation.SingleArcSimulator,
        lambert_arc_ephemeris: environment.Ephemeris,
        file_output_identifier: str,
        output_directory: str):
    """
    This function will write the results of a numerical propagation, as well as the Lambert arc states at the epochs of the
    numerical state history, to a set of files. Two files are always written when calling this function (numerical state history, a
    and Lambert arc state history). If any dependent variables are saved during the propagation, those are also saved to a file

    Parameters
    ----------
    dynamics_simulator : Object that was used to propagate the dynamics, and which contains the numerical state and dependent
                         variable results

    lambert_arc_ephemeris : Lambert arc state model as returned by the get_lambert_problem_result() function

    file_output_identifier : Name that will be used to correctly save the output data files

    output_directory : Directory to which the files will be written

    Files written
    -------------

    <output_directory/file_output_identifier>_numerical_states.dat
    <output_directory/file_output_identifier>_dependent_variables.dat
    <output_directory/file_output_identifier>_lambert_statess.dat


    Return
    ------
    None

    """

    # Save numerical states
    simulation_result = dynamics_simulator.state_history
    save2txt(solution=simulation_result,
             filename=output_directory + file_output_identifier + "_numerical_states.dat",
             directory="./")

    # Save dependent variables
    dependent_variables = dynamics_simulator.dependent_variable_history
    if len(dependent_variables.keys()) > 0:
        save2txt(solution=dependent_variables,
                 filename=output_directory + file_output_identifier + "_dependent_variables.dat",
                 directory="./")

    # Save Lambert arc states
    lambert_arc_states = get_lambert_arc_history(lambert_arc_ephemeris, simulation_result)

    save2txt(solution=lambert_arc_states,
             filename=output_directory + file_output_identifier + "_lambert_states.dat",
             directory="./")

    return

# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_problem_result(
        bodies: environment.SystemOfBodies,
        target_body: str,
        departure_epoch: float,
        arrival_epoch: float ) -> environment.Ephemeris:

    """"
    This function solved Lambert's problem for a transfer from Earth (at departure epoch) to
    a target body (at arrival epoch), with the states of Earth and the target body defined
    by ephemerides stored inside the SystemOfBodies object (bodies). Note that this solver
    assumes that the transfer departs/arrives to/from the center of mass of Earth and the target body

    Parameters
    ----------
    bodies : Body objects defining the physical simulation environment

    target_body : The name (string) of the body to which the Lambert arc is to be computed

    departure_epoch : Epoch at which the departure from Earth's center of mass is to take place

    arrival_epoch : Epoch at which the arrival at he target body's center of mass is to take place

    Return
    ------
    Ephemeris object defining a purely Keplerian trajectory. This Keplerian trajectory defines the transfer
    from Earth to the target body according to the inputs to this function. Note that this Ephemeris object
    is valid before the departure epoch, and after the arrival epoch, and simply continues (forwards and backwards)
    the unperturbed Sun-centered orbit, as fully defined by the unperturbed transfer arc
    """

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body("Sun").gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name="Sun",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=departure_epoch)

    final_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=target_body,
        observer_body_name="Sun",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=arrival_epoch)

    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_gravitational_parameter);

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(lambert_arc_initial_state,
                                                                       central_body_gravitational_parameter)

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(lambert_arc_keplerian_elements, departure_epoch,
                                              central_body_gravitational_parameter), "")

    return kepler_ephemeris

# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_arc_history(
        lambert_arc_ephemeris: environment.Ephemeris,
        simulation_result: dict ) -> dict:
    """"
    This function extracts the state history (as a dict with time as keys, and Cartesian states as values)
    from an Ephemeris object defined by a lambert solver. This function takes a dictionary of states (simulation_result)
    as input, iterates over the keys of this dict (which represent times) to ensure that the times
    at which this function returns the states of the lambert arcs are identical to those at which the
    simulation_result has (numerically calculated) states


    Parameters
    ----------
    lambert_arc_ephemeris : Ephemeris object from which the states are to be extracted

    simulation_result : Dictionary of (numerically propagated) states, from which the keys
                        are used to determine the times at which this funcion is to extract states
                        from the lambert arc
    Return
    ------
    Variational equations solver object, from which the state-, state transition matrix-, and
    sensitivity matrix history can be extracted.
    """

    lambert_arc_states = dict()
    for state in simulation_result:
        lambert_arc_states[state] = lambert_arc_ephemeris.cartesian_state(state)

    return lambert_arc_states

# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_trajectory(
        initial_time: float,
        final_time: float,
        bodies: environment.SystemOfBodies,
        lambert_arc_ephemeris: environment.Ephemeris,
        use_perturbations: bool,
        initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
        use_rsw_acceleration = False,
        rsw_acceleration_magnitude = np.array([0,0,0])) -> numerical_simulation.SingleArcSimulator:

    """
    This function will be repeatedly called throughout the assignment. Propagates the trajectory based
    on several input parameters

    Parameters
    ----------
    initial_time : Epoch since J2000 at which the propagation starts

    final_time : Epoch since J2000 at which the propagation will be terminated

    bodies : Body objects defining the physical simulation environment

    lambert_arc_ephemeris : Lambert arc state model as returned by the get_lambert_problem_result() function

    use_perturbations : Boolean to indicate whether a perturbed (True) or unperturbed (False) trajectory
                        is propagated

    initial_state_correction : Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.
    Return
    ------
    Dynamics simulator object from which the state- and dependent variable history can be extracted

    """

    # Compute initial state along Lambert arc (and apply correction if needed)
    lambert_arc_initial_state = lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction

    # Get propagator settings for perturbed/unperturbed forwards/backwards arcs
    if use_perturbations:
        propagator_settings = get_perturbed_propagator_settings(
            bodies, lambert_arc_initial_state, initial_time, final_time,use_rsw_acceleration,rsw_acceleration_magnitude)

    else:
        propagator_settings = get_unperturbed_propagator_settings(
            bodies, lambert_arc_initial_state, initial_time, final_time)

    # Propagate dynamics with required settings
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings)

    return dynamics_simulator

# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_variational_equations(
        initial_time: float,
        final_time: float,
        bodies: environment.SystemOfBodies,
        lambert_arc_ephemeris: environment.Ephemeris,
        initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
        use_rsw_acceleration = False,
        rsw_acceleration_magnitude = np.array([0,0,0])) -> numerical_simulation.SingleArcVariationalSimulator:
    """
    Propagates the variational equations for a given range of epochs for a perturbed trajectory.

    Parameters
    ----------
    initial_time : Epoch since J2000 at which the propagation starts

    final_time : Epoch since J2000 at which the propagation will be terminated

    bodies : Body objects defining the physical simulation environment

    lambert_arc_ephemeris : Lambert arc state model as returned by the get_lambert_problem_result() function

    initial_state_correction : Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.
    Return
    ------
    Variational equations solver object, from which the state-, state transition matrix-, and
    sensitivity matrix history can be extracted.
    """

    # Compute initial state along Lambert arc
    lambert_arc_initial_state = lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction

    # Get propagator settings
    propagator_settings = get_perturbed_propagator_settings(
        bodies,
        lambert_arc_initial_state,
        initial_time,
        final_time,
        use_rsw_acceleration,
        rsw_acceleration_magnitude)

    # Define parameters for variational equations
    sensitivity_parameters = get_sensitivity_parameter_set(propagator_settings, bodies, use_rsw_acceleration)

    # Propagate variational equations
    variational_equations_solver = numerical_simulation.create_variational_equations_solver(
        bodies, propagator_settings, sensitivity_parameters)

    return variational_equations_solver

# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_sensitivity_parameter_set(
        propagator_settings: propagation_setup.propagator.PropagatorSettings,
        bodies: environment.SystemOfBodies,
        use_rsw_acceleration = False) -> numerical_simulation.estimation.EstimatableParameterSet:
    """
    Function creating the parameters for which the variational equations are to be solved.

    Parameters
    ----------
    propagator_settings : Settings used for the propagation of the dynamics

    bodies : Body objects defining the physical simulation environment

    use_rsw_acceleration : Boolean denoting whether the sensitivity to an RSW acceleration is to be
                           included. Note that this can only be used (set to True) is the acceleration models
                           in propagator_settings contain an empirical acceleration


    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """
    parameter_settings = estimation_setup.parameter.initial_states(
        propagator_settings, bodies)

    if use_rsw_acceleration == True:
        parameter_settings.append(estimation_setup.parameter.constant_empirical_acceleration_terms("Spacecraft", "Sun"))

    return estimation_setup.create_parameter_set(parameter_settings, bodies, propagator_settings)


################ HELPER FUNCTIONS: MODIFY ########################################

# STUDENT CODE TASK REMOVE - full function (except signature and return)
def get_unperturbed_propagator_settings(
        bodies: environment.SystemOfBodies,
        initial_state: np.array,
        initial_time: float,
        final_time: float ) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for an unperturbed trajectory.

    Parameters
    ----------
    bodies : Body objects defining the physical simulation environment

    initial_state : Cartesian initial state of the vehicle in the simulation

    initial_time : Epoch since J2000 at which the propagation starts

    final_time : Epoch since J2000 at which the propagation will be terminated

    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """
    # Define bodies that are propagated, and their central bodies of propagation.
    bodies_to_propagate = ['Spacecraft']
    central_bodies = ['Sun']

    # Define accelerations acting on vehicle.
    acceleration_settings_on_vehicle = dict(
        Sun=[propagation_setup.acceleration.point_mass_gravity()]
    )

    # Create global accelerations dictionary.
    acceleration_settings = {'Spacecraft': acceleration_settings_on_vehicle}

    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies)
    

    # Define required outputs
    #dependent_variables_to_save = [propagation_setup.dependent_variable.cartesian_state('Spacecraft')]

    # Create numerical integrator settings.
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        fixed_step_size
    )

    initial_state_2D = np.reshape(initial_state, (6,-1))

    #dependent_variables_to_save = [propagation_setup.dependent_variable.keplerian_state('Spacecraft','Sun')]

    # Create propagation settings.
    termination_settings = propagation_setup.propagator.time_termination(final_time)
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state_2D,
        initial_time,
        integrator_settings,
        termination_settings
    )

    propagator_settings.print_settings.print_initial_and_final_conditions = True

    #propagator_settings = XXXX

    return propagator_settings

# STUDENT CODE TASK REMOVE - full function (except signature and return)
def get_perturbed_propagator_settings(
        bodies: environment.SystemOfBodies,
        initial_state: np.array,
        initial_time : float,
        final_time: float,
        use_rsw_acceleration = False,
        rsw_acceleration_magnitude = np.array([0,0,0])) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for a perturbed trajectory.

    Parameters
    ----------
    bodies : Body objects defining the physical simulation environment

    initial_state : Cartesian initial state of the vehicle in the simulation

    initial_time : Epoch since J2000 at which the propagation starts

    final_time : Epoch since J2000 at which the propagation will be terminated

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.

    Return
    ------
    Propagation settings of the perturbed trajectory.
    """
    central_bodies = ['Sun']
    bodies_to_propagate = ['Spacecraft']

    initial_state_2D = np.reshape(initial_state, (6,-1))

    # Create radiation pressure settings, and add to vehicle
    reference_area_radiation = 20.0
    radiation_pressure_coefficient = 1.2
    occulting_bodies = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient
    )
    environment_setup.add_radiation_pressure_interface(
        bodies, "Spacecraft", radiation_pressure_settings)


    # Define accelerations acting on vehicle.
    termination_settings = propagation_setup.propagator.time_termination(final_time)
    acceleration_settings_on_spacecraft = dict(
        Sun=[propagation_setup.acceleration.point_mass_gravity(),
             propagation_setup.acceleration.cannonball_radiation_pressure()],
        Venus=[propagation_setup.acceleration.point_mass_gravity()],
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Moon=[propagation_setup.acceleration.point_mass_gravity()],
        Mars=[propagation_setup.acceleration.point_mass_gravity()],
        Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
        Saturn=[propagation_setup.acceleration.point_mass_gravity()]
    )

    # Create global accelerations dictionary.
    acceleration_settings = {'Spacecraft': acceleration_settings_on_spacecraft}

    
    

    # Define required outputs
    #dependent_variables_to_save = [propagation_setup.dependent_variable.cartesian_state('Spacecraft')]

    

    # DO NOT MODIFY, and keep AFTER creation of acceleration_settings_on_spacecraft
    # (line is added for compatibility with question 4)
    if use_rsw_acceleration:
        acceleration_settings_on_spacecraft["Sun"].append(
            propagation_setup.acceleration.empirical(rsw_acceleration_magnitude))
        
    # Create acceleration models.
    acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies)
        
    fixed_step_size = 3600.0

    # If propagation is backwards in time, make initial time step negative
    if initial_time > final_time:
        signed_fixed_step_size = -fixed_step_size
    else:
        signed_fixed_step_size = fixed_step_size

    # Create numerical integrator settings.
    
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        signed_fixed_step_size
    )

    # Define required outputs
    dependent_variables_to_save = [propagation_setup.dependent_variable.total_acceleration('Spacecraft')]

    # Create propagation settings.
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state_2D,
        initial_time,
        integrator_settings,
        termination_settings,
        output_variables = dependent_variables_to_save
    )

    return propagator_settings

# STUDENT CODE TASK REMOVE - full function (except signature and return)
def create_simulation_bodies( ) -> environment.SystemOfBodies:

    """
    Creates the body objects required for the simulation, using the
    environment_setup.create_system_of_bodies for natural bodies,
    and manual definition for vehicles

    Parameters
    ----------
    none

    Return
    ------
    Body objects required for the simulation.

    """

    bodies_to_create = ['Sun','Venus','Earth','Moon','Mars','Jupiter','Saturn']
    global_frame_origin = 'Sun'
    global_frame_orientation = 'ECLIPJ2000'
    body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.create_empty_body( 'Spacecraft' )
    bodies.get("Spacecraft").mass = 1000.0

    #bodies = XXXX

    return bodies


def computeError(initial_variation,arc_index,arc_length,bodies,lambert_arc_ephemeris):

    lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, (departure_epoch), (arrival_epoch))

    # Compute relevant parameters (dynamics, state transition matrix, Delta V) for each arc
    
    # Compute initial and final time for arc
    current_arc_initial_time = departure_epoch + 62*3600 + arc_index*arc_length
    current_arc_final_time = current_arc_initial_time + arc_length

    ###########################################################################
    # RUN CODE FOR QUESTION 3a ################################################
    ###########################################################################

    ###########################################################################
    # RUN CODE FOR QUESTION 3c/d/e ############################################
    ###########################################################################
    # Note: for question 3e, part of the code below will be put into a loop
    # for the requested iterations

    # Solve for state transition matrix on current arc
    variational_equations_solver = propagate_variational_equations(current_arc_initial_time,
                                                                current_arc_final_time, bodies,
                                                                lambert_arc_ephemeris)
    state_transition_matrix_history = variational_equations_solver.state_transition_matrix_history
    state_history = variational_equations_solver.state_history
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    # Get final state transition matrix (and its inverse)
    initial_epoch = list(state_transition_matrix_history.keys())[0]
    final_epoch = list(state_transition_matrix_history.keys())[-1]
    final_state_transition_matrix = state_transition_matrix_history[final_epoch]
    state_transition_matrix3x3 = final_state_transition_matrix[:3,3:]
    inverse_state_transition_matrix3x3 = np.linalg.inv(state_transition_matrix3x3)

    # Retrieve final state deviation
    final_state_deviation = state_history[final_epoch] - lambert_history[final_epoch]

    # Compute required velocity change at beginning of arc to meet required final state

    deltaV = inverse_state_transition_matrix3x3 @ final_state_deviation[:3].T
    initial_state_correction = -np.concatenate([np.array([0,0,0]),deltaV])

    error = np.zeros([len(state_transition_matrix_history),6])

    for i in range(len(state_transition_matrix_history)):

        epoch = list(state_transition_matrix_history.keys())[i]
        state_transition_matrix = state_transition_matrix_history[epoch]

        error_t = state_transition_matrix@initial_state_correction - state_history[epoch] + lambert_history[epoch]
        error[i,:] = error_t

    errorTotal = np.linalg.norm(error,axis=1)

    return errorTotal




    # Propagate with correction to initial state (use propagate_trajecory function),
    # and its optional initial_state_correction input
    dynamics_simulator = propagate_trajectory(current_arc_initial_time, current_arc_final_time, bodies, lambert_arc_ephemeris,
                            use_perturbations = True, initial_state_correction=initial_state_correction)

        



