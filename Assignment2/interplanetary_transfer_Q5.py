''' 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

from interplanetary_transfer_helper_functions import *
import matplotlib.pyplot as plt

# Load spice kernels.
spice_interface.load_standard_kernels( )

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 5 #################################################
###########################################################################

# Create body objects
bodies = create_simulation_bodies()

# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)


# Set arc length
number_of_arcs = 10
arc_length = arc_length = (time_of_flight-450000)/number_of_arcs

limitArray = np.zeros([10,6])



for arc_index in range(number_of_arcs):

    # Compute start and end time for current arc
    current_arc_initial_time = departure_epoch + 62*3600 + arc_index*arc_length
    current_arc_final_time = current_arc_initial_time + arc_length
    # Get propagator settings for perturbed forward arc
    arc_initial_state = lambert_arc_ephemeris.cartesian_state(current_arc_initial_time)
    propagator_settings = get_perturbed_propagator_settings(bodies, arc_initial_state, current_arc_initial_time, current_arc_final_time)

    ###########################################################################
    # PROPAGATE NOMINAL TRAJECTORY AND VARIATIONAL EQUATIONS ##################
    ###########################################################################
    sensitivity_parameters = get_sensitivity_parameter_set(
        propagator_settings, bodies, target_body)
    variational_equations_simulator = numerical_simulation.create_variational_equations_solver(
        bodies, propagator_settings, sensitivity_parameters)

    state_transition_result = variational_equations_simulator.state_transition_matrix_history
    nominal_integration_result = variational_equations_simulator.state_history

    

    for i in range(6):

        initial_state_change = np.array([0,0,0,0,0,0])

        if i<=2:
            initial_state_change[i] = 1e8

        elif i>2:
            initial_state_change[i] = 1e5

        converged = False
        max_position_previous = 0
        max_position_previous = 0
        increment = 0.5 * initial_state_change

        while converged == False:

            error = computeError(initial_state_change,nominal_integration_result,state_transition_result,0,arc_length,bodies,lambert_arc_ephemeris,current_arc_initial_time,current_arc_final_time)

            #maxPositionErrors = np.array([max(error[:,0]),max(error[:,1]),max(error[:,2])])
            maxPositionError = max(np.linalg.norm(error[:,0:3],axis=1))
            #maxVelocityErrors = np.array([max(error[:,3]),max(error[:,4]),max(error[:,5])])
            maxVelocityError = max(np.linalg.norm(error[:,3:],axis=1))

            #maxDifferencePosition = abs(maxPositionError - max_position_previous)
            #maxDifferenceVelocity = abs(maxVelocityError-max_velocity_previous)

            if max(increment) < 0.01:
            #if abs(maxPositionError - max_position_previous)<1e2 and abs(maxVelocityError-max_velocity_previous)<0.00001:

                converged=True
                limitArray[arc_index,i] = initial_state_change[i]

            max_position_previous = maxPositionError
            max_velocity_previous = maxVelocityError


            if maxPositionError > 1e8 and maxVelocityError > 1:

                initial_state_change = initial_state_change - increment
                increment = 0.5*increment

            elif maxPositionError < 1e8 and maxVelocityError >1:

                initial_state_change = initial_state_change - increment
                increment = 0.5*increment

            elif maxPositionError > 1e8 and maxVelocityError <1:

                initial_state_change = initial_state_change - increment
                increment = 0.5*increment

            elif maxPositionError < 1e8 and maxVelocityError <1:

                initial_state_change = initial_state_change + increment
                increment = 0.5*increment


limitArrayRSW = np.zeros([10,6])

for arc_index in range(number_of_arcs):

    # Compute start and end time for current arc
    current_arc_initial_time = departure_epoch + 62*3600 + arc_index*arc_length
    current_arc_final_time = current_arc_initial_time + arc_length
    # Get propagator settings for perturbed forward arc
    arc_initial_state = lambert_arc_ephemeris.cartesian_state(current_arc_initial_time)
    propagator_settings = get_perturbed_propagator_settings(bodies, arc_initial_state, current_arc_initial_time, current_arc_final_time)

    ###########################################################################
    # PROPAGATE NOMINAL TRAJECTORY AND VARIATIONAL EQUATIONS ##################
    ###########################################################################
    sensitivity_parameters = get_sensitivity_parameter_set(
        propagator_settings, bodies, target_body)
    variational_equations_simulator = numerical_simulation.create_variational_equations_solver(
        bodies, propagator_settings, sensitivity_parameters)

    state_transition_result = variational_equations_simulator.state_transition_matrix_history
    nominal_integration_result = variational_equations_simulator.state_history

    

    for i in range(6):

        initial_state_change = np.array([0,0,0,0,0,0])

        if i<=2:
            initial_state_change[i] = 1e8

        elif i>2:
            initial_state_change[i] = 1e5

        converged = False
        max_position_previous = 0
        max_position_previous = 0
        increment = 0.5 * initial_state_change

        while converged == False:

            error = computeErrorRSW(initial_state_change,nominal_integration_result,state_transition_result,0,arc_length,bodies,lambert_arc_ephemeris,current_arc_initial_time,current_arc_final_time)

             #maxPositionErrors = np.array([max(error[:,0]),max(error[:,1]),max(error[:,2])])
            maxPositionError = max(np.linalg.norm(error[:,0:3],axis=1))
            #maxVelocityErrors = np.array([max(error[:,3]),max(error[:,4]),max(error[:,5])])
            maxVelocityError = max(np.linalg.norm(error[:,3:],axis=1))

            #maxDifferencePosition = abs(maxPositionError - max_position_previous)
            #maxDifferenceVelocity = abs(maxVelocityError-max_velocity_previous)

            #if abs(maxPositionError - max_position_previous)<1e4 and abs(maxVelocityError-max_velocity_previous)<0.01:
            if max(increment) < 0.01:

                converged=True
                limitArrayRSW[arc_index,i] = initial_state_change[i]

            max_position_previous = maxPositionError
            max_velocity_previous = maxVelocityError


            if maxPositionError > 1e8 and maxVelocityError > 1:

                initial_state_change = initial_state_change - increment
                increment = 0.5*increment

            elif maxPositionError < 1e8 and maxVelocityError >1:

                initial_state_change = initial_state_change - increment
                increment = 0.5*increment

            elif maxPositionError > 1e8 and maxVelocityError <1:

                initial_state_change = initial_state_change - increment
                increment = 0.5*increment

            elif maxPositionError < 1e8 and maxVelocityError <1:

                initial_state_change = initial_state_change + increment
                increment = 0.5*increment    


i = np.array([1,2,3,4,5,6,7,8,9,10])
rNorm = np.linalg.norm(limitArray[:,:3],axis=1)
rNormRSW = np.linalg.norm(limitArrayRSW[:,:3],axis=1)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(i,rNorm)
ax1.grid()
ax1.set_xlabel('Arc')
ax1.set_ylabel('Normalised limit value (position)')
ax1.set_title(r'Limit values of $\Delta x_{0}$ in the inertial frame')
ax1.set_xlim(1,10)

fig2 = plt.figure()
ax1 = fig2.add_subplot(1,1,1)
ax1.plot(i,rNormRSW)
ax1.grid()
ax1.set_xlabel('Arc')
ax1.set_ylabel('Normalised limit value (position)')
ax1.set_title(r'Limit values of $\Delta x_{0}$ in the RSW frame')
ax1.set_xlim(1,10)


plt.show()

np.savetxt('Question5_Results_AE4868_2023_2_4659554.dat', limitArray)



