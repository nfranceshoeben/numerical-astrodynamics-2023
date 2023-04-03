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
# RUN CODE FOR QUESTION 4 #################################################
###########################################################################

rsw_acceleration_magnitude = [0, 0, 0]

# Create body objects
bodies = create_simulation_bodies()

# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)

###########################################################################
# RUN CODE FOR QUESTION 4b ################################################
###########################################################################

# Set start and end times of full trajectory
departure_epoch_with_buffer = departure_epoch + 62*3600
arrival_epoch_with_buffer = arrival_epoch - 63*3600

# Solve for state transition matrix on current arc
variational_equations_solver = propagate_variational_equations(
    departure_epoch_with_buffer,
    arrival_epoch_with_buffer,
    bodies,
    lambert_arc_ephemeris,
    use_rsw_acceleration = True,
    rsw_acceleration_magnitude=rsw_acceleration_magnitude)

sensitivity_matrix_history = variational_equations_solver.sensitivity_matrix_history
state_history = variational_equations_solver.state_history
lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

# Get final state transition matrix (and its inverse)
initial_epoch = list(sensitivity_matrix_history.keys())[0]
final_epoch = list(sensitivity_matrix_history.keys())[-1]

# Compute the difference between the Lambert solution and the numerical one:

final_state_deviation = state_history[final_epoch] - lambert_history[final_epoch]

# Retrieve final sensitivity matrix

final_sensitivity_matrix = sensitivity_matrix_history[final_epoch]

# Finding thrust parameter

thrustParameter = np.linalg.solve(final_sensitivity_matrix[:3,:],-final_state_deviation[:3])

#thrustParameter = np.linalg.inv(final_sensitivity_matrix[:3,3:]) @ final_state_deviation

# Compute low-thrust RSW acceleration to meet required final position
#rsw_acceleration_magnitude = np.linalg.norm(thrustParameter)

# Propagate dynamics with RSW acceleration. NOTE: use the rsw_acceleration_magnitude as
# input to the propagate_trajectory function
dynamics_simulator = propagate_trajectory(departure_epoch_with_buffer, arrival_epoch_with_buffer, bodies, 
                                          lambert_arc_ephemeris, use_perturbations=True, use_rsw_acceleration=True,
                                          rsw_acceleration_magnitude=thrustParameter)

########################## Plotting test ##################################

array = result2array(dynamics_simulator.state_history)
lambert = result2array(get_lambert_arc_history(lambert_arc_ephemeris,dynamics_simulator.state_history))

difference = array-lambert

print(f'Last difference; {difference[-1,:]}')

lastArray = array[-1,:]
lastLambert = lambert[-1,:]

ax = plt.figure().add_subplot(projection='3d')
ax.plot(array[:,1],array[:,2],array[:,3])
ax.plot(lambert[:,1],lambert[:,2],lambert[:,3])

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
ax2 = fig1.add_subplot(1,3,2)
ax3 = fig1.add_subplot(1,3,3)

ax1.plot(array[:,0],(array[:,1]-lambert[:,1]))
ax2.plot(array[:,0],(array[:,2]-lambert[:,2]))
ax3.plot(array[:,0],(array[:,3]-lambert[:,3]))

fig2 = plt.figure()
ax1 = fig2.add_subplot(1,3,1)
ax2 = fig2.add_subplot(1,3,2)
ax3 = fig2.add_subplot(1,3,3)

ax1.plot(array[:,0],(array[:,4]-lambert[:,4]))
ax2.plot(array[:,0],(array[:,5]-lambert[:,5]))
ax3.plot(array[:,0],(array[:,6]-lambert[:,6]))

print(f'Final epoch: {final_epoch}')

plt.show()

###########################################################################
# RUN CODE FOR QUESTION 4d ################################################
###########################################################################

number_of_arcs = 2
arc_length = (time_of_flight-450000)/number_of_arcs
rsw_acceleration_magnitude = [0, -1e-8, 1e-9]

# Compute arc 1: ----------------------------------------------------------
arc_start_epoch = departure_epoch_with_buffer
arc_end_epoch = departure_epoch_with_buffer + arc_length

variational_equations_solver = propagate_variational_equations(
    arc_start_epoch,
    arc_end_epoch,
    bodies,
    lambert_arc_ephemeris,
    use_rsw_acceleration = True,
    rsw_acceleration_magnitude=rsw_acceleration_magnitude)

sensitivity_matrix_history = variational_equations_solver.sensitivity_matrix_history
state_history = variational_equations_solver.state_history
lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

# Get final state transition matrix (and its inverse)
initial_epoch = list(sensitivity_matrix_history.keys())[0]
final_epoch = list(sensitivity_matrix_history.keys())[-1]

# Compute the difference between the Lambert solution and the numerical one:

final_state_deviation = state_history[final_epoch] - lambert_history[final_epoch]

# Retrieve final sensitivity matrix

final_sensitivity_matrix_1 = sensitivity_matrix_history[final_epoch]

# Finding thrust parameter

thrustParameter1 = np.linalg.solve(final_sensitivity_matrix_1[:3,:],-final_state_deviation[:3])

# Propagate dynamics with RSW acceleration. NOTE: use the rsw_acceleration_magnitude as
# input to the propagate_trajectory function
dynamics_simulator = propagate_trajectory(arc_start_epoch, arc_end_epoch, bodies, 
                                        lambert_arc_ephemeris, use_perturbations=True, use_rsw_acceleration=True,
                                        rsw_acceleration_magnitude=rsw_acceleration_magnitude)

write_propagation_results_to_file(
            dynamics_simulator, lambert_arc_ephemeris, "Q4_arc_1",output_directory+'/Q4_Arcs/')

array = result2array(dynamics_simulator.state_history)
lambert = result2array(get_lambert_arc_history(lambert_arc_ephemeris,dynamics_simulator.state_history))

difference = array-lambert


# Compute arc 2 -----------------------------------------------------------

arc_start_epoch = arc_end_epoch
arc_end_epoch = arc_start_epoch + arc_length


variational_equations_solver = propagate_variational_equations(
    arc_start_epoch,
    arc_end_epoch,
    bodies,
    lambert_arc_ephemeris,
    initial_state_correction=difference[-1,1:],
    use_rsw_acceleration = True,
    rsw_acceleration_magnitude=rsw_acceleration_magnitude)

sensitivity_matrix_history = variational_equations_solver.sensitivity_matrix_history
state_transition_matrix_history = variational_equations_solver.state_transition_matrix_history
state_history = variational_equations_solver.state_history
lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

# Get final state transition matrix (and its inverse)
initial_epoch = list(sensitivity_matrix_history.keys())[0]
final_epoch = list(sensitivity_matrix_history.keys())[-1]

# Compute the difference between the Lambert solution and the numerical one:

final_state_deviation = state_history[final_epoch] - lambert_history[final_epoch]

# Retrieve final sensitivity matrix

final_sensitivity_matrix_2 = sensitivity_matrix_history[final_epoch]
final_state_transition_matrix_2 = state_transition_matrix_history[final_epoch]

# Finding thrust parameter

thrustParameter2 = np.linalg.solve(final_sensitivity_matrix_2[:3,:],
                                   (-final_state_deviation[:3]))

#thrustParameter = np.linalg.inv(final_sensitivity_matrix[:3,3:]) @ final_state_deviation

# Compute low-thrust RSW acceleration to meet required final position
#rsw_acceleration_magnitude = np.linalg.norm(thrustParameter)

# Propagate dynamics with RSW acceleration. NOTE: use the rsw_acceleration_magnitude as
# input to the propagate_trajectory function
dynamics_simulator = propagate_trajectory(arc_start_epoch, arc_end_epoch, bodies, 
                                        lambert_arc_ephemeris, use_perturbations=True,initial_state_correction=difference[-1,1:], use_rsw_acceleration=True,
                                        rsw_acceleration_magnitude=thrustParameter2)

write_propagation_results_to_file(
            dynamics_simulator, lambert_arc_ephemeris, "Q4_arc_2",output_directory+'/Q4_Arcs/')

# ------------------------------------- Plot -------------------------------------------------

arc1numerical = np.genfromtxt('./SimulationOutput/Q4_Arcs/Q4_arc_1_numerical_states.dat')
arc2numerical = np.genfromtxt('./SimulationOutput/Q4_Arcs/Q4_arc_2_numerical_states.dat')


arc1lambert = np.genfromtxt('./SimulationOutput/Q4_Arcs/Q4_arc_1_lambert_states.dat')
arc2lambert = np.genfromtxt('./SimulationOutput/Q4_Arcs/Q4_arc_2_lambert_states.dat')

numerical = np.concatenate((arc1numerical,arc2numerical))
lambert = np.concatenate((arc1lambert,arc2lambert))

difference = numerical-lambert
print(f'The final difference is: {difference[-1,:]}')

fig1 = plt.figure()

ax1 = fig1.add_subplot(1,3,1)
ax2 = fig1.add_subplot(1,3,2)
ax3 = fig1.add_subplot(1,3,3)

ax1.plot(numerical[:,0],(numerical[:,1]-lambert[:,1]))
ax2.plot(numerical[:,0],(numerical[:,2]-lambert[:,2]))
ax3.plot(numerical[:,0],(numerical[:,3]-lambert[:,3]))

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')

ax1.set_ylabel('Position difference [m]')
ax2.set_ylabel('Position difference [m]')
ax3.set_ylabel('Position difference [m]')

ax1.set_title('Position difference in X-direction')
ax2.set_title('Position difference in Y-direction')
ax3.set_title('Position difference in Z-direction')

ax1.set_xlim(min(numerical[:,0]),max(numerical[:,0]))
ax2.set_xlim(min(numerical[:,0]),max(numerical[:,0]))
ax3.set_xlim(min(numerical[:,0]),max(numerical[:,0]))

fig2 = plt.figure()

ax1 = fig2.add_subplot(1,3,1)
ax2 = fig2.add_subplot(1,3,2)
ax3 = fig2.add_subplot(1,3,3)

ax1.plot(numerical[:,0],(numerical[:,4]-lambert[:,4]))
ax2.plot(numerical[:,0],(numerical[:,5]-lambert[:,5]))
ax3.plot(numerical[:,0],(numerical[:,6]-lambert[:,6]))

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')

ax1.set_ylabel('Velocity difference [m]')
ax2.set_ylabel('Velocity difference [m]')
ax3.set_ylabel('Velocity difference [m]')

ax1.set_title('Velocity difference in X-direction')
ax2.set_title('Velocity difference in Y-direction')
ax3.set_title('Velocity difference in Z-direction')

ax1.set_xlim(min(numerical[:,0]),max(numerical[:,0]))
ax2.set_xlim(min(numerical[:,0]),max(numerical[:,0]))
ax3.set_xlim(min(numerical[:,0]),max(numerical[:,0]))

ax = plt.figure().add_subplot(projection='3d')
ax.plot(numerical[:,1],numerical[:,2],numerical[:,3])
ax.plot(lambert[:,1],lambert[:,2],lambert[:,3])

plt.show()

###########################################################################
# RUN CODE FOR QUESTION 4e ################################################
###########################################################################

number_of_arcs = 2
arc_length = (time_of_flight-450000)/number_of_arcs
rsw_acceleration_magnitude = [0, -1e-8, 1e-9]

# Compute arc 1: ----------------------------------------------------------
arc_start_epoch = departure_epoch_with_buffer
arc_end_epoch = departure_epoch_with_buffer + arc_length

variational_equations_solver = propagate_variational_equations(
    arc_start_epoch,
    arc_end_epoch,
    bodies,
    lambert_arc_ephemeris,
    use_rsw_acceleration = True,
    rsw_acceleration_magnitude=rsw_acceleration_magnitude)

sensitivity_matrix_history = variational_equations_solver.sensitivity_matrix_history
state_history = variational_equations_solver.state_history
lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

# Get final state transition matrix (and its inverse)
initial_epoch = list(sensitivity_matrix_history.keys())[0]
final_epoch = list(sensitivity_matrix_history.keys())[-1]

# Compute the difference between the Lambert solution and the numerical one:

final_state_deviation = state_history[final_epoch] - lambert_history[final_epoch]

# Retrieve final sensitivity matrix

final_sensitivity_matrix_1 = sensitivity_matrix_history[final_epoch]

# Finding thrust parameter

thrustParameter1 = np.linalg.solve(final_sensitivity_matrix_1[:3,:],-final_state_deviation[:3])

# Propagate dynamics with RSW acceleration. NOTE: use the rsw_acceleration_magnitude as
# input to the propagate_trajectory function
dynamics_simulator = propagate_trajectory(arc_start_epoch, arc_end_epoch, bodies, 
                                        lambert_arc_ephemeris, use_perturbations=True, use_rsw_acceleration=True,
                                        rsw_acceleration_magnitude=rsw_acceleration_magnitude)

write_propagation_results_to_file(
            dynamics_simulator, lambert_arc_ephemeris, "Q4e_arc_1",output_directory+'/Q4_Arcs/')

array = result2array(dynamics_simulator.state_history)
lambert = result2array(get_lambert_arc_history(lambert_arc_ephemeris,dynamics_simulator.state_history))

difference = array-lambert


# Compute arc 2 -----------------------------------------------------------

arc_start_epoch = arc_end_epoch
arc_end_epoch = arc_start_epoch + arc_length

converged = False
correction = np.array([0,0,0])

thrustParameterIterative = np.array([0,0,0])
iterations = 0

while converged==False:

    iterations += 1

    variational_equations_solver = propagate_variational_equations(
        arc_start_epoch,
        arc_end_epoch,
        bodies,
        lambert_arc_ephemeris,
        initial_state_correction=difference[-1,1:],
        use_rsw_acceleration = True,
        rsw_acceleration_magnitude=thrustParameterIterative)

    sensitivity_matrix_history = variational_equations_solver.sensitivity_matrix_history
    state_transition_matrix_history = variational_equations_solver.state_transition_matrix_history
    state_history = variational_equations_solver.state_history
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    # Get final state transition matrix (and its inverse)
    initial_epoch = list(sensitivity_matrix_history.keys())[0]
    final_epoch = list(sensitivity_matrix_history.keys())[-1]

    # Compute the difference between the Lambert solution and the numerical one:

    final_state_deviation = state_history[final_epoch] - lambert_history[final_epoch]

    # Retrieve final sensitivity matrix

    final_sensitivity_matrix_2 = sensitivity_matrix_history[final_epoch]
    final_state_transition_matrix_2 = state_transition_matrix_history[final_epoch]

    positionSensitivityMatrix = final_sensitivity_matrix_2[0:3,0:3]

    # Finding thrust parameter

    #thrustParameter2 = np.linalg.solve(final_sensitivity_matrix_2[:3,:],
                                    #(-final_state_deviation[:3]))
                                
    thrustParameter2 = -np.linalg.inv(positionSensitivityMatrix) @ final_state_deviation[0:3]
    
    
    thrustParameterIterative = thrustParameterIterative + thrustParameter2

    #thrustParameter = np.linalg.inv(final_sensitivity_matrix[:3,3:]) @ final_state_deviation

    # Compute low-thrust RSW acceleration to meet required final position
    #rsw_acceleration_magnitude = np.linalg.norm(thrustParameter)

    # Propagate dynamics with RSW acceleration. NOTE: use the rsw_acceleration_magnitude as
    # input to the propagate_trajectory function
    dynamics_simulator = propagate_trajectory(arc_start_epoch, arc_end_epoch, bodies, 
                                            lambert_arc_ephemeris, use_perturbations=True,initial_state_correction=difference[-1,1:], use_rsw_acceleration=True,
                                            rsw_acceleration_magnitude=thrustParameterIterative)

    array = result2array(dynamics_simulator.state_history)
    #array = result2array(state_history)
    lambert = result2array(get_lambert_arc_history(lambert_arc_ephemeris,dynamics_simulator.state_history))

    error = array[-1,:3]-lambert[-1,:3]
    errorMag = np.linalg.norm(error[:3])

    if errorMag < 1:
        converged=True
        results = dynamics_simulator

write_propagation_results_to_file(
            dynamics_simulator, lambert_arc_ephemeris, "Q4e_arc_2",output_directory+'/Q4_Arcs/')

array = result2array(dynamics_simulator.state_history)
lambert = result2array(get_lambert_arc_history(lambert_arc_ephemeris,dynamics_simulator.state_history))


arc1numerical = np.genfromtxt('./SimulationOutput/Q4_Arcs/Q4e_arc_1_numerical_states.dat')
arc2numerical = np.genfromtxt('./SimulationOutput/Q4_Arcs/Q4e_arc_2_numerical_states.dat')


arc1lambert = np.genfromtxt('./SimulationOutput/Q4_Arcs/Q4e_arc_1_lambert_states.dat')
arc2lambert = np.genfromtxt('./SimulationOutput/Q4_Arcs/Q4e_arc_2_lambert_states.dat')

numerical = np.concatenate((arc1numerical,arc2numerical))
lambert = np.concatenate((arc1lambert,arc2lambert))


print(f'The number of iterations is {iterations}')

difference = numerical-lambert
print(f'The final difference is: {difference[-1,:]}')

fig1 = plt.figure()

ax1 = fig1.add_subplot(1,3,1)
ax2 = fig1.add_subplot(1,3,2)
ax3 = fig1.add_subplot(1,3,3)

ax1.plot(numerical[:,0],(numerical[:,1]-lambert[:,1]))
ax2.plot(numerical[:,0],(numerical[:,2]-lambert[:,2]))
ax3.plot(numerical[:,0],(numerical[:,3]-lambert[:,3]))

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')

ax1.set_ylabel('Position difference [m]')
ax2.set_ylabel('Position difference [m]')
ax3.set_ylabel('Position difference [m]')

ax1.set_title('Position difference in X-direction')
ax2.set_title('Position difference in Y-direction')
ax3.set_title('Position difference in Z-direction')

ax1.set_xlim(min(numerical[:,0]),max(numerical[:,0]))
ax2.set_xlim(min(numerical[:,0]),max(numerical[:,0]))
ax3.set_xlim(min(numerical[:,0]),max(numerical[:,0]))

fig2 = plt.figure()

ax1 = fig2.add_subplot(1,3,1)
ax2 = fig2.add_subplot(1,3,2)
ax3 = fig2.add_subplot(1,3,3)

ax1.plot(numerical[:,0],(numerical[:,4]-lambert[:,4]))
ax2.plot(numerical[:,0],(numerical[:,5]-lambert[:,5]))
ax3.plot(numerical[:,0],(numerical[:,6]-lambert[:,6]))

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xlabel('Epoch [s]')
ax2.set_xlabel('Epoch [s]')
ax3.set_xlabel('Epoch [s]')

ax1.set_ylabel('Velocity difference [m]')
ax2.set_ylabel('Velocity difference [m]')
ax3.set_ylabel('Velocity difference [m]')

ax1.set_title('Velocity difference in X-direction')
ax2.set_title('Velocity difference in Y-direction')
ax3.set_title('Velocity difference in Z-direction')

ax1.set_xlim(min(numerical[:,0]),max(numerical[:,0]))
ax2.set_xlim(min(numerical[:,0]),max(numerical[:,0]))
ax3.set_xlim(min(numerical[:,0]),max(numerical[:,0]))

ax = plt.figure().add_subplot(projection='3d')
ax.plot(numerical[:,1],numerical[:,2],numerical[:,3])
ax.plot(lambert[:,1],lambert[:,2],lambert[:,3])

plt.show()
