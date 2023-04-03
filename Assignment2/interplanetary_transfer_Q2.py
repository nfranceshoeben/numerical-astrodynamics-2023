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
from tudatpy.kernel.interface import spice
from tudatpy.kernel.interface import spice_interface
import matplotlib.pyplot as plt

# Load spice kernels.
spice_interface.load_standard_kernels( )

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 2 #################################################
###########################################################################

# Create body objects
bodies = create_simulation_bodies()

# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)

"""
case_i: The initial and final propagation time equal to the initial and final times of the Lambert arc.
case_ii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour.
case_iii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=2 days.

"""

soiEarth = 9.24527e8
soiMars = 5.76232e8

lambertData = np.genfromtxt('./SimulationOutput/Q2a_0_lambert_states.dat')

for i in range(len(lambertData[:,0])):

    earthPos = spice.get_body_cartesian_state_at_epoch(
        target_body_name='Earth',
        observer_body_name='Sun',
        reference_frame_name='ECLIPJ2000',
        aberration_corrections='NONE',
        ephemeris_time=lambertData[i,0]
    )

    difference = lambertData[i,1:4] - earthPos[0:3]
    distance = np.linalg.norm(difference)

    if distance >= soiEarth:
        soiEarthDeparture = i
        break

for i in range(1,len(lambertData[:,0])):

    marsPos = spice.get_body_cartesian_state_at_epoch(
        target_body_name='Mars',
        observer_body_name='Sun',
        reference_frame_name='ECLIPJ2000',
        aberration_corrections='NONE',
        ephemeris_time=lambertData[-i,0]
    )

    difference = lambertData[-i,1:4] - marsPos[0:3]
    distance = np.linalg.norm(difference)

    if distance >= soiMars:
        soiMarsDeparture = len(lambertData[:,0])-i
        break

soiEarthDepartureEpoch = lambertData[soiEarthDeparture,0]
soiMarsDepartureEpoch = lambertData[soiMarsDeparture,0]

print(f'Departing Earth at epoch {soiEarthDepartureEpoch}, epoch number {soiEarthDeparture}.')
print(f'Arriving around Mars at epoch {soiMarsDepartureEpoch}, epoch number {soiMarsDeparture}')
print(f'Total epochs: {len(lambertData[:,0])}')









# List cases to iterate over. STUDENT NOTE: feel free to modify if you see fit
cases = ['case_i', 'case_ii', 'case_iii']

run = True

if run:

    # Run propagation for each of cases i-iii
    for case in cases:

        
        # Define the initial and final propagation time for the current case
        if case == 'case_i':
            departure_epoch_with_buffer = departure_epoch
            arrival_epoch_with_buffer = arrival_epoch

        if case == 'case_ii':
            departure_epoch_with_buffer = departure_epoch + 3600
            arrival_epoch_with_buffer = arrival_epoch - 3600

        if case == 'case_iii':
            departure_epoch_with_buffer = soiEarthDepartureEpoch
            arrival_epoch_with_buffer = soiMarsDepartureEpoch


        # Perform propagation
        dynamics_simulator = propagate_trajectory( departure_epoch_with_buffer, arrival_epoch_with_buffer, bodies, lambert_arc_ephemeris,
                            use_perturbations = True)
        write_propagation_results_to_file(
            dynamics_simulator, lambert_arc_ephemeris, "Q2a_" + str(cases.index(case)), output_directory)

        state_history = dynamics_simulator.state_history
        lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

################################## PART 2 ######################################

epoch_midpoint = departure_epoch + (arrival_epoch - departure_epoch)/2

run2 = True

if run2:

    dynamics_simulator_2_1 = propagate_trajectory(epoch_midpoint,departure_epoch, bodies, lambert_arc_ephemeris, use_perturbations=True)
    write_propagation_results_to_file(
                dynamics_simulator_2_1, lambert_arc_ephemeris, "Q2b_1", output_directory)

    dynamics_simulator_2_2 = propagate_trajectory(epoch_midpoint,arrival_epoch, bodies, lambert_arc_ephemeris, use_perturbations=True)
    write_propagation_results_to_file(
                dynamics_simulator_2_2, lambert_arc_ephemeris, "Q2b_2", output_directory)



################################## Plotting ######################################

lambertA0 = np.genfromtxt('./SimulationOutput/Q2a_0_lambert_states.dat')
numericalA0 = np.genfromtxt('./SimulationOutput/Q2a_0_numerical_states.dat')
accelerationsA0 = np.genfromtxt('./SimulationOutput/Q2a_0_dependent_variables.dat')

lambertA1 = np.genfromtxt('./SimulationOutput/Q2a_1_lambert_states.dat')
numericalA1 = np.genfromtxt('./SimulationOutput/Q2a_1_numerical_states.dat')
accelerationsA1 = np.genfromtxt('./SimulationOutput/Q2a_1_dependent_variables.dat')

lambertA2 = np.genfromtxt('./SimulationOutput/Q2a_2_lambert_states.dat')
numericalA2 = np.genfromtxt('./SimulationOutput/Q2a_2_numerical_states.dat')
accelerationsA2 = np.genfromtxt('./SimulationOutput/Q2a_2_dependent_variables.dat')

lambertB1 = np.genfromtxt('./SimulationOutput/Q2b_1_lambert_states.dat')
numericalB1 = np.genfromtxt('./SimulationOutput/Q2b_1_numerical_states.dat')
accelerationsB1 = np.genfromtxt('./SimulationOutput/Q2b_1_dependent_variables.dat')

lambertB2 = np.genfromtxt('./SimulationOutput/Q2b_2_lambert_states.dat')
numericalB2 = np.genfromtxt('./SimulationOutput/Q2b_2_numerical_states.dat')
accelerationsB2 = np.genfromtxt('./SimulationOutput/Q2b_2_dependent_variables.dat')

lambertB = np.concatenate((lambertB1,lambertB2),axis=0)
numericalB = np.concatenate((numericalB1,numericalB2),axis=0)
accelerationsB = np.concatenate((accelerationsB1,accelerationsB2),axis=0)

lambertAcceleration0x = 1.327e20 / (lambertA0[:,1])**2
lambertAcceleration0y = 1.327e20 / (lambertA0[:,2])**2
lambertAcceleration0z = 1.327e20 / (lambertA0[:,3])**2

lambertAcceleration1x = 1.327e20 / (lambertA1[:,1])**2
lambertAcceleration1y = 1.327e20 / (lambertA1[:,2])**2
lambertAcceleration1z = 1.327e20 / (lambertA1[:,3])**2

lambertAcceleration2x = 1.327e20 / (lambertA2[:,1])**2
lambertAcceleration2y = 1.327e20 / (lambertA2[:,2])**2
lambertAcceleration2z = 1.327e20 / (lambertA2[:,3])**2

lambertAccelerationBx = 1.327e20 / (lambertB[:,1])**2
lambertAccelerationBy = 1.327e20 / (lambertB[:,2])**2
lambertAccelerationBz = 1.327e20 / (lambertB[:,3])**2

lambertAcceleration0 = np.vstack((lambertAcceleration0x,lambertAcceleration0y,lambertAcceleration0z))
lambertAcceleration1 = np.vstack((lambertAcceleration1x,lambertAcceleration1y,lambertAcceleration1z))
lambertAcceleration2 = np.vstack((lambertAcceleration2x,lambertAcceleration2y,lambertAcceleration2z))
lambertAccelerationB = np.vstack((lambertAccelerationBx,lambertAccelerationBy,lambertAccelerationBz))

accelerationA0Delta = accelerationsA0[:,1:] - lambertAcceleration0.T
accelerationA1Delta = accelerationsA1[:,1:] - lambertAcceleration1.T
accelerationA2Delta = accelerationsA2[:,1:] - lambertAcceleration2.T
accelerationBDelta = accelerationsB[:,1:] - lambertAccelerationB.T

A0Delta = numericalA0 - lambertA0
A1Delta = numericalA1 - lambertA1
A2Delta = numericalA2 - lambertA2
BDelta = numericalB - lambertB

A0Position = np.linalg.norm(A0Delta[:,1:4],axis=1)
A1Position = np.linalg.norm(A1Delta[:,1:4],axis=1)
A2Position = np.linalg.norm(A2Delta[:,1:4],axis=1)
BPosition = np.linalg.norm(BDelta[:,1:4],axis=1)

A0Velocity = np.linalg.norm(A0Delta[:,3:],axis=1)
A1Velocity = np.linalg.norm(A1Delta[:,3:],axis=1)
A2Velocity = np.linalg.norm(A2Delta[:,3:],axis=1)
BVelocity = np.linalg.norm(BDelta[:,3:],axis=1)

A0Acceleration = np.linalg.norm(A0Delta,axis=1)
A1Acceleration = np.linalg.norm(A1Delta,axis=1)
A2Acceleration = np.linalg.norm(A2Delta,axis=1)
BAcceleration = np.linalg.norm(BDelta,axis=1)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
ax2 = fig1.add_subplot(1,3,2)
ax3 = fig1.add_subplot(1,3,3)
ax1.plot(lambertA0[:,0],A0Position)
ax2.plot(lambertA0[:,0],A0Velocity)
ax3.plot(lambertA0[:,0],A0Acceleration)
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_title('Difference in position')
ax2.set_title('Difference in velocity')
ax3.set_title('Difference in acceleration')
ax1.set_xlabel('Epoch')
ax2.set_xlabel('Epoch')
ax3.set_xlabel('Epoch')
ax1.set_ylabel('Difference [$m$]')
ax2.set_ylabel('Difference [$m/s$]')
ax3.set_ylabel('Difference [$m/s^2$]')
ax1.set_xlim(min(lambertA0[:,0]),max(lambertA0[:,0]))
ax2.set_xlim(min(lambertA0[:,0]),max(lambertA0[:,0]))
ax3.set_xlim(min(lambertA0[:,0]),max(lambertA0[:,0]))

fig2 = plt.figure()
ax1 = fig2.add_subplot(1,3,1)
ax2 = fig2.add_subplot(1,3,2)
ax3 = fig2.add_subplot(1,3,3)
ax1.plot(lambertA1[:,0],A1Position)
ax2.plot(lambertA1[:,0],A1Velocity)
ax3.plot(lambertA1[:,0],A1Acceleration)
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_title('Difference in position')
ax2.set_title('Difference in velocity')
ax3.set_title('Difference in acceleration')
ax1.set_xlabel('Epoch')
ax2.set_xlabel('Epoch')
ax3.set_xlabel('Epoch')
ax1.set_ylabel(r'Difference [$m$]')
ax2.set_ylabel(r'Difference [$m/s$]')
ax3.set_ylabel(r'Difference [$m/s^2$]')
ax1.set_xlim(min(lambertA1[:,0]),max(lambertA1[:,0]))
ax2.set_xlim(min(lambertA1[:,0]),max(lambertA1[:,0]))
ax3.set_xlim(min(lambertA1[:,0]),max(lambertA1[:,0]))

fig3 = plt.figure()
ax1 = fig3.add_subplot(1,3,1)
ax2 = fig3.add_subplot(1,3,2)
ax3 = fig3.add_subplot(1,3,3)
ax1.plot(lambertA2[:,0],A2Position)
ax2.plot(lambertA2[:,0],A2Velocity)
ax3.plot(lambertA2[:,0],A2Acceleration)
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_title('Difference in position')
ax2.set_title('Difference in velocity')
ax3.set_title('Difference in acceleration')
ax1.set_xlabel('Epoch')
ax2.set_xlabel('Epoch')
ax3.set_xlabel('Epoch')
ax1.set_ylabel(r'Difference [$m$]')
ax2.set_ylabel(r'Difference [$m/s$]')
ax3.set_ylabel(r'Difference [$m/s^2$]')
ax1.set_xlim(min(lambertA2[:,0]),max(lambertA2[:,0]))
ax2.set_xlim(min(lambertA2[:,0]),max(lambertA2[:,0]))
ax3.set_xlim(min(lambertA2[:,0]),max(lambertA2[:,0]))

fig4 = plt.figure()
ax1 = fig4.add_subplot(1,3,1)
ax2 = fig4.add_subplot(1,3,2)
ax3 = fig4.add_subplot(1,3,3)
ax1.plot(lambertB[:,0],BPosition)
ax2.plot(lambertB[:,0],BVelocity)
ax3.plot(lambertB[:,0],BAcceleration)
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_title('Difference in position')
ax2.set_title('Difference in velocity')
ax3.set_title('Difference in acceleration')
ax1.set_xlabel('Epoch')
ax2.set_xlabel('Epoch')
ax3.set_xlabel('Epoch')
ax1.set_ylabel(r'Difference [$m$]')
ax2.set_ylabel(r'Difference [$m/s$]')
ax3.set_ylabel(r'Difference [$m/s^2$]')
ax1.set_xlim(min(lambertB[:,0]),max(lambertB[:,0]))
ax2.set_xlim(min(lambertB[:,0]),max(lambertB[:,0]))
ax3.set_xlim(min(lambertB[:,0]),max(lambertB[:,0]))

plt.show()