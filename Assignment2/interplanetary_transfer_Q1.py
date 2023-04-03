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

# Run switch

run = True

###########################################################################
# RUN CODE FOR QUESTION 1 #################################################
###########################################################################

if run:

    # Create body objects
    bodies = create_simulation_bodies( )

    target_body = 'Mars'
    #departure_epoch = departure_epoch
    #arrival_epoch = departure_epoch + 215.3577652*86400


    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)

    # Create propagation settings and propagate dynamics
    dynamics_simulator = propagate_trajectory( departure_epoch, arrival_epoch, bodies, lambert_arc_ephemeris,
                        use_perturbations = False)

    # Write results to file
    write_propagation_results_to_file(
        dynamics_simulator, lambert_arc_ephemeris, "Q1",output_directory)

    # Extract state history from dynamics simulator
    state_history = dynamics_simulator.state_history

    # Evaluate the Lambert arc model at each of the epochs in the state_history
    lambert_history = get_lambert_arc_history( lambert_arc_ephemeris, state_history )


# Plot

lambertData = np.genfromtxt('./SimulationOutput/Q1_lambert_states.dat')
numericalData = np.genfromtxt('./SimulationOutput/Q1_numerical_states.dat')

ax = plt.figure().add_subplot(projection='3d')
ax.plot(numericalData[:,1]/1e9,numericalData[:,2]/1e9,numericalData[:,3]/1e9,'k', label='Numerical Trajectory')
#ax.plot(lambertData[:,1],lambertData[:,2],lambertData[:,3],'r', label='Lambert Trajectory')
ax.scatter(0,0,0,c='C1',label='Sun',s=20)
ax.scatter(lambertData[0,1]/1e9,lambertData[0,2]/1e9,lambertData[0,3]/1e9,c='C0',label='Earth (roughly)',s=10)
ax.scatter(lambertData[-1,1]/1e9,lambertData[-1,2]/1e9,lambertData[-1,3]/1e9,c='r',label='Mars (roughly)',s=8)
#ax.set_aspect('equal')
ax.set_xlabel(r'X-direction [$10^9$ m]')
ax.set_ylabel(r'Y-direction [$10^9$ m]')
ax.set_zlabel(r'Z-direction [$10^9$ m]')
ax.legend()

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,3,1)
ax1.plot(numericalData[:,0],numericalData[:,1]-lambertData[:,1])
ax1.set_xlabel('Epoch [s]')
ax1.set_ylabel('Difference [m]')
ax1.set_title("Position difference x-direction")
ax1.grid()

ax2 = fig1.add_subplot(1,3,2)
ax2.plot(numericalData[:,0],numericalData[:,2]-lambertData[:,2])
ax2.set_xlabel('Epoch [s]')
ax2.set_ylabel('Difference [m]')
ax2.set_title("Position difference y-direction")
ax2.grid()

ax3 = fig1.add_subplot(1,3,3)
ax3.plot(numericalData[:,0],numericalData[:,3]-lambertData[:,3])
ax3.set_xlabel('Epoch [s]')
ax3.set_ylabel('Difference [m]')
ax3.set_title("Position difference z-direction")
ax3.grid()



plt.show()
