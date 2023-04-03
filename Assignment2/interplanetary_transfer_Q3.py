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
from tudatpy.util import result2array

# Load spice kernels.
spice_interface.load_standard_kernels( )

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 3 #################################################
###########################################################################

# Create body objects
bodies = create_simulation_bodies()

# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, (departure_epoch), (arrival_epoch))


##############################################################
# Compute number of arcs and arc length
number_of_arcs = 10
arc_length = (time_of_flight-450000)/number_of_arcs

##############################################################

run = True
dV = 0
previous_velocity = 0

#q3a_deviation = np.array([[0],[0],[0],[0],[0],[0]])
q3a_deviation = np.array([0,0,0,0,0,0])

if run:
# Compute relevant parameters (dynamics, state transition matrix, Delta V) for each arc
    for arc_index in range(number_of_arcs):

        # Compute initial and final time for arc
        current_arc_initial_time = departure_epoch + 62*3600 + arc_index*arc_length
        current_arc_final_time = current_arc_initial_time + arc_length

        ###########################################################################
        # RUN CODE FOR QUESTION 3a ################################################
        ###########################################################################

        # Propagate dynamics on current arc (use propagate_trajecory function)
        dynamics_simulator = propagate_trajectory(current_arc_initial_time, current_arc_final_time, bodies, lambert_arc_ephemeris,
                                use_perturbations = True)
        
        dynamics_simulator_a = propagate_trajectory(current_arc_initial_time, current_arc_final_time, bodies, lambert_arc_ephemeris,
                                use_perturbations = True, initial_state_correction=q3a_deviation)

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
        sensitivity_matrix_history = variational_equations_solver.sensitivity_matrix_history
        state_history = variational_equations_solver.state_history
        lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

        # Get final state transition matrix (and its inverse)
        initial_epoch = list(state_transition_matrix_history.keys())[0]
        final_epoch = list(state_transition_matrix_history.keys())[-1]
        final_state_transition_matrix = state_transition_matrix_history[final_epoch]
        state_transition_matrix3x3 = final_state_transition_matrix[:3,3:]
        inverse_state_transition_matrix3x3 = np.linalg.inv(state_transition_matrix3x3)

        # Retrieve initial state deviation
        initial_state_deviation = state_history[initial_epoch] - lambert_history[initial_epoch]

        # Retrieve final state deviation
        final_state_deviation = state_history[final_epoch] - lambert_history[final_epoch]

        if arc_index == 0:
            lambert_history_1 = lambert_history

        # Compute required velocity change at beginning of arc to meet required final state

        deltaV = inverse_state_transition_matrix3x3 @ final_state_deviation[:3].T
        initial_state_correction = -np.concatenate([np.array([0,0,0]),deltaV])

        # Propagate with correction to initial state (use propagate_trajecory function),
        # and its optional initial_state_correction input
        dynamics_simulator = propagate_trajectory(current_arc_initial_time, current_arc_final_time, bodies, lambert_arc_ephemeris,
                                use_perturbations = True, initial_state_correction=initial_state_correction)
        
        #data = np.zeros([int(time_of_flight/fixed_step_size),7])

        # Write results to file
        write_propagation_results_to_file(
            dynamics_simulator, lambert_arc_ephemeris, f"Q3_arc_{arc_index+1}",output_directory+'/Q3_Arcs/')
        
        

        q3a_deviation = result2array(dynamics_simulator_a.state_history)[-1,1:] - result2array(get_lambert_arc_history(lambert_arc_ephemeris,dynamics_simulator_a.state_history))[-1,1:]

        write_propagation_results_to_file(
            dynamics_simulator_a, lambert_arc_ephemeris,f"Q3a_arc_{arc_index+1}",output_directory+'/Q3_Arcs/'
        )
        
        state_history = dynamics_simulator.state_history
        #finalResult = result2array(dynamics_simulator)[-1,3:]
        finalVelDifference = (state_history[final_epoch] - lambert_history[final_epoch])[3:]
        finalVelDifferenceTotal = np.linalg.norm(finalVelDifference)
        initialVelDifference = np.linalg.norm(deltaV)

        
        
        dV += np.linalg.norm(deltaV) + finalVelDifferenceTotal

print(f'The total Delta V is: {dV} [m/s]')

arc1Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_1_numerical_states.dat')
arc2Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_2_numerical_states.dat')
arc3Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_3_numerical_states.dat')
arc4Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_4_numerical_states.dat')
arc5Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_5_numerical_states.dat')
arc6Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_6_numerical_states.dat')
arc7Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_7_numerical_states.dat')
arc8Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_8_numerical_states.dat')
arc9Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_9_numerical_states.dat')
arc10Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_10_numerical_states.dat')

arc1Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_1_lambert_states.dat')
arc2Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_2_lambert_states.dat')
arc3Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_3_lambert_states.dat')
arc4Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_4_lambert_states.dat')
arc5Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_5_lambert_states.dat')
arc6Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_6_lambert_states.dat')
arc7Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_7_lambert_states.dat')
arc8Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_8_lambert_states.dat')
arc9Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_9_lambert_states.dat')
arc10Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3a_arc_10_lambert_states.dat')

numericalStates = np.vstack((arc1Numerical,arc2Numerical,arc3Numerical,arc4Numerical,arc5Numerical,arc6Numerical,arc7Numerical,arc8Numerical,arc9Numerical,arc10Numerical))
LambertStates = np.vstack((arc1Lambert,arc2Lambert,arc3Lambert,arc4Lambert,arc5Lambert,arc6Lambert,arc7Lambert,arc8Lambert,arc9Lambert,arc10Lambert))

fig0 = plt.figure()
ax1 = fig0.add_subplot(1,3,1)
ax2 = fig0.add_subplot(1,3,2)
ax3 = fig0.add_subplot(1,3,3)
ax1.plot(LambertStates[:,0],(numericalStates[:,1]-LambertStates[:,1]))
ax2.plot(LambertStates[:,0],(numericalStates[:,2]-LambertStates[:,2]))
ax3.plot(LambertStates[:,0],(numericalStates[:,3]-LambertStates[:,3]))
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')
ax1.set_ylabel('Position difference [m]')
ax2.set_ylabel('Position difference [m]')
ax3.set_ylabel('Position difference [m]')
ax1.set_title('X-direction position error')
ax2.set_title('Y-direction position error')
ax3.set_title('Z-direction position error')
ax1.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax2.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax3.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))

fig02 = plt.figure()
ax1 = fig02.add_subplot(1,3,1)
ax2 = fig02.add_subplot(1,3,2)
ax3 = fig02.add_subplot(1,3,3)
ax1.plot(LambertStates[:,0],(numericalStates[:,4]-LambertStates[:,4]))
ax2.plot(LambertStates[:,0],(numericalStates[:,5]-LambertStates[:,5]))
ax3.plot(LambertStates[:,0],(numericalStates[:,6]-LambertStates[:,6]))
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')
ax1.set_ylabel('Velocity difference [m]')
ax2.set_ylabel('Velocity difference [m]')
ax3.set_ylabel('Velocity difference [m]')
ax1.set_title('X-direction Velocity error')
ax2.set_title('Y-direction Velocity error')
ax3.set_title('Z-direction Velocity error')
ax1.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax2.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax3.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))

plt.show()


arc1Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_1_numerical_states.dat')
arc2Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_2_numerical_states.dat')
arc3Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_3_numerical_states.dat')
arc4Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_4_numerical_states.dat')
arc5Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_5_numerical_states.dat')
arc6Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_6_numerical_states.dat')
arc7Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_7_numerical_states.dat')
arc8Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_8_numerical_states.dat')
arc9Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_9_numerical_states.dat')
arc10Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_10_numerical_states.dat')

arc1Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_1_lambert_states.dat')
arc2Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_2_lambert_states.dat')
arc3Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_3_lambert_states.dat')
arc4Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_4_lambert_states.dat')
arc5Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_5_lambert_states.dat')
arc6Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_6_lambert_states.dat')
arc7Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_7_lambert_states.dat')
arc8Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_8_lambert_states.dat')
arc9Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_9_lambert_states.dat')
arc10Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_arc_10_lambert_states.dat')

numericalStates = np.vstack((arc1Numerical,arc2Numerical,arc3Numerical,arc4Numerical,arc5Numerical,arc6Numerical,arc7Numerical,arc8Numerical,arc9Numerical,arc10Numerical))
LambertStates = np.vstack((arc1Lambert,arc2Lambert,arc3Lambert,arc4Lambert,arc5Lambert,arc6Lambert,arc7Lambert,arc8Lambert,arc9Lambert,arc10Lambert))

ax = plt.figure().add_subplot(projection='3d')
ax.plot(numericalStates[:,1],numericalStates[:,2],numericalStates[:,3])
ax.plot(LambertStates[:,1],LambertStates[:,2],LambertStates[:,3])

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
ax2 = fig1.add_subplot(1,3,2)
ax3 = fig1.add_subplot(1,3,3)
ax1.plot(LambertStates[:,0],(numericalStates[:,1]-LambertStates[:,1]))
ax2.plot(LambertStates[:,0],(numericalStates[:,2]-LambertStates[:,2]))
ax3.plot(LambertStates[:,0],(numericalStates[:,3]-LambertStates[:,3]))
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')
ax1.set_ylabel('Position difference [m]')
ax2.set_ylabel('Position difference [m]')
ax3.set_ylabel('Position difference [m]')
ax1.set_title('X-direction position error')
ax2.set_title('Y-direction position error')
ax3.set_title('Z-direction position error')
ax1.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax2.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax3.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))

fig2 = plt.figure()
ax1 = fig2.add_subplot(1,3,1)
ax2 = fig2.add_subplot(1,3,2)
ax3 = fig2.add_subplot(1,3,3)
ax1.plot(LambertStates[:,0],(numericalStates[:,4]-LambertStates[:,4]))
ax2.plot(LambertStates[:,0],(numericalStates[:,5]-LambertStates[:,5]))
ax3.plot(LambertStates[:,0],(numericalStates[:,6]-LambertStates[:,6]))
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')
ax1.set_ylabel('Velocity difference [m/s]')
ax2.set_ylabel('Velocity difference [m/s]')
ax3.set_ylabel('Velocity difference [m/s]')
ax1.set_title('X-direction velocity error')
ax2.set_title('Y-direction velocity error')
ax3.set_title('Z-direction velocity error')
ax1.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax2.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax3.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))

plt.show()



####################################### PART E ############################################

run2 = True

if run2:

    arc_num = []
# Compute relevant parameters (dynamics, state transition matrix, Delta V) for each arc
    for arc_index in range(number_of_arcs):

        # Compute initial and final time for arc
        current_arc_initial_time = departure_epoch + 62*3600 + arc_index*arc_length
        current_arc_final_time = current_arc_initial_time + arc_length

        ###########################################################################
        # RUN CODE FOR QUESTION 3a ################################################
        ###########################################################################

        # Propagate dynamics on current arc (use propagate_trajecory function)
        dynamics_simulator = propagate_trajectory(current_arc_initial_time, current_arc_final_time, bodies, lambert_arc_ephemeris,
                                use_perturbations = True)

        ###########################################################################
        # RUN CODE FOR QUESTION 3c/d/e ############################################
        ###########################################################################
        # Note: for question 3e, part of the code below will be put into a loop
        # for the requested iterations

        # Solve for state transition matrix on current arc

        check = True
        results = variational_equations_solver.state_history
        simulation_results = lambert_arc_ephemeris
        delta_isc = results[final_epoch] - lambert_history[final_epoch]
        iterations = 0



        while check == True:

            iterations += 1

            variational_equations_solver = propagate_variational_equations(current_arc_initial_time,
                                                                        current_arc_final_time, bodies,
                                                                        lambert_arc_ephemeris,
                                                                        initial_state_correction=initial_state_correction)
            
            state_transition_matrix_history = variational_equations_solver.state_transition_matrix_history

            state_history = variational_equations_solver.state_history
            lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

            

            # Get final state transition matrix (and its inverse)
            final_epoch = list(state_transition_matrix_history.keys())[-1]
            final_state_transition_matrix = state_transition_matrix_history[final_epoch]
            state_transition_matrix3x3 = final_state_transition_matrix[:3,3:]
            inverse_state_transition_matrix3x3 = np.linalg.inv(state_transition_matrix3x3)
            results = variational_equations_solver.state_history

            # Retrieve final state deviation
            final_state_deviation = results[final_epoch] - lambert_history[final_epoch]
            # Compute required velocity change at beginning of arc to meet required final state

            deltaV = inverse_state_transition_matrix3x3 @ final_state_deviation[:3].T
            delta_isc = -np.concatenate([np.array([0,0,0]),deltaV]) #LA FÒRMULA NO S'EQUIVOCA - ETS TU QUE T'EQUIVOQUES
            #initial_state_correction = np.concatenate([np.array([0,0,0]),deltaV])

            # Propagate with correction to initial state (use propagate_trajecory function),
            # and its optional initial_state_correction input
           
            

            write_propagation_results_to_file(
            dynamics_simulator, lambert_arc_ephemeris, f"Q3_e_temp",output_directory+'/Q3_Arcs/')

            results_lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_temp_lambert_states.dat')

            
            difference = list(results.values())[-1] - list(lambert_history.values())[-1]

            #difference = result2array(results - lambert_history)

            differenceTotal = np.linalg.norm(difference[0:3])

            check = False
            initial_state_correction += delta_isc

            '''for i in range(len(differenceTotal)):

                if differenceTotal[i] > 1:
                    check=True'''
            if differenceTotal > 1:
                check=True
                simulation_results = results

            

        arc_num.append(iterations)
        dynamics_simulator = propagate_trajectory(current_arc_initial_time, current_arc_final_time, bodies, lambert_arc_ephemeris,
                                use_perturbations = True, initial_state_correction=initial_state_correction)
                
        
        #data = np.zeros([int(time_of_flight/fixed_step_size),7])

        # Write results to file
        write_propagation_results_to_file(
            dynamics_simulator, lambert_arc_ephemeris, f"Q3_e_arc_{arc_index+1}",output_directory+'/Q3_Arcs/')
        
print(f'The number of iterations per arc is {arc_num}')
        
arc1Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_1_numerical_states.dat')
arc2Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_2_numerical_states.dat')
arc3Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_3_numerical_states.dat')
arc4Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_4_numerical_states.dat')
arc5Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_5_numerical_states.dat')
arc6Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_6_numerical_states.dat')
arc7Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_7_numerical_states.dat')
arc8Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_8_numerical_states.dat')
arc9Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_9_numerical_states.dat')
arc10Numerical = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_10_numerical_states.dat')

arc1Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_1_lambert_states.dat')
arc2Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_2_lambert_states.dat')
arc3Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_3_lambert_states.dat')
arc4Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_4_lambert_states.dat')
arc5Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_5_lambert_states.dat')
arc6Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_6_lambert_states.dat')
arc7Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_7_lambert_states.dat')
arc8Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_8_lambert_states.dat')
arc9Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_9_lambert_states.dat')
arc10Lambert = np.genfromtxt('./SimulationOutput/Q3_Arcs/Q3_e_arc_10_lambert_states.dat')

numericalStatesE = np.vstack((arc1Numerical,arc2Numerical,arc3Numerical,arc4Numerical,arc5Numerical,arc6Numerical,arc7Numerical,arc8Numerical,arc9Numerical,arc10Numerical))
LambertStatesE = np.vstack((arc1Lambert,arc2Lambert,arc3Lambert,arc4Lambert,arc5Lambert,arc6Lambert,arc7Lambert,arc8Lambert,arc9Lambert,arc10Lambert))

ax = plt.figure().add_subplot(projection='3d')
ax.plot(numericalStatesE[:,1],numericalStatesE[:,2],numericalStatesE[:,3])
ax.plot(LambertStatesE[:,1],LambertStatesE[:,2],LambertStatesE[:,3])

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
ax2 = fig1.add_subplot(1,3,2)
ax3 = fig1.add_subplot(1,3,3)
ax1.plot(LambertStatesE[:,0],(numericalStatesE[:,1]-LambertStatesE[:,1]))
ax2.plot(LambertStatesE[:,0],(numericalStatesE[:,2]-LambertStatesE[:,2]))
ax3.plot(LambertStatesE[:,0],(numericalStatesE[:,3]-LambertStatesE[:,3]))
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')
ax1.set_ylabel('Position difference [m]')
ax2.set_ylabel('Position difference [m]')
ax3.set_ylabel('Position difference [m]')
ax1.set_title('X-direction position error')
ax2.set_title('Y-direction position error')
ax3.set_title('Z-direction position error')
ax1.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax2.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax3.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))

fig2 = plt.figure()
ax1 = fig2.add_subplot(1,3,1)
ax2 = fig2.add_subplot(1,3,2)
ax3 = fig2.add_subplot(1,3,3)
ax1.plot(LambertStatesE[:,0],(numericalStatesE[:,4]-LambertStatesE[:,4]))
ax2.plot(LambertStatesE[:,0],(numericalStatesE[:,5]-LambertStatesE[:,5]))
ax3.plot(LambertStatesE[:,0],(numericalStatesE[:,6]-LambertStatesE[:,6]))
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')
ax1.set_ylabel('Velocity difference [m/s]')
ax2.set_ylabel('Velocity difference [m/s]')
ax3.set_ylabel('Velocity difference [m/s]')
ax1.set_title('X-direction velocity error')
ax2.set_title('Y-direction velocity error')
ax3.set_title('Z-direction velocity error')
ax1.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax2.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))
ax3.set_xlim(min(LambertStates[:,0]),max(LambertStates[:,0]))

plt.show()