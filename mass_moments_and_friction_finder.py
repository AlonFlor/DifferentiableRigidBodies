import file_handling
import draw_data
import os
import numpy as np
import scipy

motion_script = file_handling.read_motion_script_file(os.path.join("test1","motion_script.csv"))
dt = 0.001

print(motion_script[0])

velocities = []
external_force_magnitudes = []
for time_step_line in motion_script:
    external_force_magn = float(time_step_line[-5])
    external_force_location = np.array([float(time_step_line[-4]), float(time_step_line[-3]), float(time_step_line[-2])])
    external_force_direction = np.array([float(time_step_line[-1]), 0., 0.])
    if external_force_magn != 0:
        velocity = np.array([float(time_step_line[8]), float(time_step_line[9]), float(time_step_line[10])])
        velocity_magn = np.linalg.norm(velocity)

        cos_theta=None
        if velocity_magn==0:
            cos_theta = 1
        else:
            cos_theta = np.dot(external_force_direction, velocity/velocity_magn)

        external_force_magnitudes.append(external_force_magn*cos_theta)
        velocities.append(velocity_magn)


delta_velocities_divided_by_dt = []
for i in np.arange(len(velocities)-1):
    delta_velocities_divided_by_dt.append((velocities[i+1] - velocities[i])/dt)

draw_data.plot_data(external_force_magnitudes[:-1], delta_velocities_divided_by_dt)


draw_data.plot_data(external_force_magnitudes[100:-1], delta_velocities_divided_by_dt[100:])

velocity_regression_result = scipy.stats.linregress(external_force_magnitudes[100:-1], delta_velocities_divided_by_dt[100:])

print("slope:", velocity_regression_result.slope)
print("intercept:", velocity_regression_result.intercept)

print("mass result:", 1./velocity_regression_result.slope)
print("translational mu result:", velocity_regression_result.intercept/-9.8)




'''
Main problem seems to be that friction does not oppose the force, it opposes the current velocity, which may definitely be turned.

Trying just the x direction gives me the correct slope before friction acts, but friction varies with the time step due to direction changes.
Trying the total velocity would give me constant friction (need to check that), but would require adjusting the force.
'''
#print(motion_script[-1][-6:])

#thingy = motion_script[-1]
#print(thingy[-6:])