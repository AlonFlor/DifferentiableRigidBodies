import pybullet as p
import numpy as np
import draw_data
import scipy

import time

physicsClient = p.connect(p.DIRECT)
#physicsClient = p.connect(p.GUI)

p.setGravity(0,0,-9.8)

objectID = p.loadURDF("urdf and meshes\hammer.urdf")



startPos = [0,0,0.5]
startOrientation = p.getQuaternionFromEuler([np.pi/2,0,0])
p.resetBasePositionAndOrientation(objectID, startPos, startOrientation)

planeID = p.loadURDF("urdf and meshes\plane.urdf")
print("plane:",planeID)
print("object:",objectID)
#p.changeDynamics(planeID, -1, mass=0)

dynamics = p.getDynamicsInfo(objectID, -1)
print("dynamics:",dynamics)
dynamics = p.getDynamicsInfo(planeID, -1)
print("dynamics:",dynamics)
dt = 1./240.

#time.sleep(20)

#for i in range (2400):
    #if i < 100:
def run_pybullet_one_time_step(first_force,second_force):
    #p.applyExternalForce(objectID, 0, (0., 400., 0.), (0., 0., 0.5), p.WORLD_FRAME)
    #p.applyExternalForce(objectID, 0, (0., 800., 0.), (20., 0., 0.5), p.WORLD_FRAME)
    print("forces applied:")
    print("\t",first_force[0],"\tat",first_force[1])
    print("\t",second_force[0],"\tat",second_force[1])
    p.applyExternalForce(objectID, 0, first_force[0], first_force[1], p.WORLD_FRAME)
    p.applyExternalForce(objectID, 0, second_force[0], second_force[1], p.WORLD_FRAME)

    p.stepSimulation()
    time.sleep(dt)

    location, orientation = p.getBasePositionAndOrientation(objectID)
    #print("step", i)
    print("\tlocation:\t",location)
    print("\torientation:\t", orientation)
    velocity, angular_velocity = p.getBaseVelocity(objectID)
    print("\tvelocity:\t",velocity)
    print("\tangular velocity:\t", angular_velocity)

    contacts = p.getContactPoints()
    print("\tcontact info:")
    print("\t\tnumber of contacts:",len(contacts))
    contact_made = []
    normal_forces = []
    frictions = []
    count = 0
    for info in contacts:
        if info[9] != 0:
            contact_made.append(count)#print(info[8], info[9],info[10])
            normal_forces.append(info[9])
            local_friction = np.array([info[11][0] * info[10], info[11][1] * info[10], info[11][2] * info[10]]) + \
                             np.array([info[13][0] * info[12], info[13][1] * info[12], info[13][2] * info[12]])
            frictions.append(local_friction)
        count+=1
    print("\t\t",contact_made, "have contacts")
    print("\t\tnormal forces",normal_forces)
    print("\t\tfrictions",frictions)
    total_friction = np.array([0., 0., 0.])
    for friction in frictions:
        total_friction += friction
    print("\t\ttotal friction",total_friction)

    p.resetBasePositionAndOrientation(objectID, startPos, startOrientation)
    p.resetBaseVelocity(objectID, (0., 0., 0.), (0., 0., 0.))

    return velocity, angular_velocity, total_friction

#p.setTimeStep()

def get_actual_mass_and_com():
    count = 1
    mass = 1.
    loc_weighed_mass = np.array(list(p.getBasePositionAndOrientation(objectID)[0]))
    for i in range(p.getNumJoints(objectID)):
        this_mass = 1.#p.getDynamicsInfo(objectID, -1)[0]
        this_loc = p.getLinkState(objectID, i)[0]
        mass += this_mass
        loc_weighed_mass += np.array(list(this_loc))*this_mass
        count += 1
    return mass, loc_weighed_mass/count


def try_a_force_pair(first_force, second_force):
    print("trying a force pair")
    velocity, angular_velocity, friction = run_pybullet_one_time_step(first_force, second_force)

    return velocity, angular_velocity[2], friction

def find_balancing_external_force_pair(first_force_magn, pos1, dir1, pos2, dir2):
    """Given a force, find the other force such that both forces pushing together cause no rotational motion. This requires trial and error, which will be done here via binary search"""

    force_magn_pairs_found = []
    angular_velocity_changes_found = []
    frictions = []

    #get initial left part of binary search
    second_force_magn = -1. * first_force_magn
    velocity_result, angular_velocity_result, friction = try_a_force_pair([(dir1[0]*first_force_magn, dir1[1]*first_force_magn, dir1[2]*first_force_magn), pos1],
                                                                [(dir2[0]*second_force_magn, dir2[1]*second_force_magn, dir2[2]*second_force_magn), pos2])
    left = second_force_magn, velocity_result, angular_velocity_result

    #record results for later
    frictions.append(friction)
    force_magn_pairs_found.append((first_force_magn, second_force_magn))
    angular_velocity_changes_found.append(angular_velocity_result)

    #get initial right part of binary search
    second_force_magn = 2. * first_force_magn
    velocity_result, angular_velocity_result, friction = try_a_force_pair([(dir1[0]*first_force_magn, dir1[1]*first_force_magn, dir1[2]*first_force_magn), pos1],
                                                                [(dir2[0]*second_force_magn, dir2[1]*second_force_magn, dir2[2]*second_force_magn), pos2])
    right = second_force_magn, velocity_result, angular_velocity_result

    #record results for later
    frictions.append(friction)
    force_magn_pairs_found.append((first_force_magn, second_force_magn))
    angular_velocity_changes_found.append(angular_velocity_result)

    #do the binary search
    result = left if (np.abs(left[2]) < np.abs(right[2])) else right
    count = 0
    while np.abs(result[2]) > 1e-9:  # threshold
        second_force_magn = 0.5 * (left[0] + right[0])
        velocity_result, angular_velocity_result, friction = try_a_force_pair([(dir1[0]*first_force_magn, dir1[1]*first_force_magn, dir1[2]*first_force_magn), pos1],
                                                                    [(dir2[0]*second_force_magn, dir2[1]*second_force_magn, dir2[2]*second_force_magn), pos2])
        center = second_force_magn, velocity_result, angular_velocity_result

        # record results for later
        frictions.append(friction)
        force_magn_pairs_found.append((first_force_magn, second_force_magn))
        angular_velocity_changes_found.append(angular_velocity_result)

        result = center

        if center[2] * left[2] < 0:
            right = center
        else:
            left = center
        count += 1

    return result[0], result[1], force_magn_pairs_found, angular_velocity_changes_found, frictions


def mass_moments_finder():

    #get locations
    pos1 = (0., 0., 0.5)
    dir1 = (0., 1., 0.)
    pos2 = (20., 0., 0.5)
    dir2 = (0., 1., 0.)
    external_force_contact_location_first_force = pos1
    first_force_rx = external_force_contact_location_first_force[0]
    
    external_force_contact_location_second_force = pos2
    second_force_rx = external_force_contact_location_second_force[0]
    second_to_first_force_rx = first_force_rx - second_force_rx

    #magnitudes of the force at each location
    first_force_magns = []
    second_force_magns = []

    #info for getting mass
    forces = []
    delta_velocities_divided_by_dt = []

    #info for getting moment of inertia
    force_magn_pairs = []
    angular_velocity_changes_divided_by_dt = []

    first_force_magn = 7000.
    limit = 2000
    stride = 100

    #friction info. Friction should be constant across all rounds
    frictions = []
    translation_only_motion_frictions = []

    print("start")

    for i in np.arange(0, limit, stride):
        second_force_magn, velocity_result, force_magn_pairs_found, angular_velocity_changes_found, frictions_result = \
            find_balancing_external_force_pair(first_force_magn, pos1, dir1, pos2, dir2)

        #add data for translation-only motion
        first_force_magns.append(first_force_magn)
        second_force_magns.append(second_force_magn)
        forces.append(first_force_magn + second_force_magn)
        delta_velocities_divided_by_dt.append(velocity_result[1] / dt)

        #add data for motion in general
        translation_only_motion_frictions.append(frictions_result[-1])
        frictions += frictions_result
        force_magn_pairs += force_magn_pairs_found
        for angular_velocity_change in angular_velocity_changes_found:
            angular_velocity_changes_divided_by_dt.append(angular_velocity_change / dt)
        first_force_magn += stride

    #print("force magn pairs:\n", first_force_magns[0], second_force_magns[0], "\n", first_force_magns[1], second_force_magns[1])

    #get com x coord
    com_pos_found_x_list = []
    end = len(first_force_magns) - 1
    for i in np.arange(end):
        com_to_first_force_rx = \
            second_to_first_force_rx * (second_force_magns[end] - second_force_magns[i]) / (first_force_magns[end] - first_force_magns[i] + second_force_magns[end] - second_force_magns[i])
        com_pos_found_x = first_force_rx - com_to_first_force_rx
        com_pos_found_x_list.append(com_pos_found_x)

    actual_mass, actual_com = get_actual_mass_and_com()
    actual_com_x_list = [actual_com[0]]*len(com_pos_found_x_list)

    #draw stuff
    draw_data.plot_data_two_curves(first_force_magns[1:], com_pos_found_x_list, actual_com_x_list)
    draw_data.plot_data(forces, delta_velocities_divided_by_dt)
    print("drawing friction-related stuff")
    translation_only_motion_frictions_x = []
    translation_only_motion_frictions_y = []
    translation_only_motion_frictions_magn = []
    for friction in translation_only_motion_frictions:
        translation_only_motion_frictions_x.append(friction[0])
        translation_only_motion_frictions_y.append(friction[1])
        translation_only_motion_frictions_magn.append(np.linalg.norm(friction))
    print("\tx-axis friction")
    draw_data.plot_data(first_force_magns, translation_only_motion_frictions_x)
    print("\ty-axis friction")
    draw_data.plot_data(first_force_magns, translation_only_motion_frictions_y)
    print("\tfriction magnitude")
    draw_data.plot_data(first_force_magns, translation_only_motion_frictions_magn)

    print("\n\ncenter of mass x-axis result:")
    print("\t",com_pos_found_x_list[len(com_pos_found_x_list)-1])
    print("actual value:")
    print("\t", actual_com[0])

    #get mass
    print("\n\nprocessing mass")
    velocity_regression_result = scipy.stats.linregress(forces, delta_velocities_divided_by_dt)
    print("slope:", velocity_regression_result.slope)
    print("intercept:", velocity_regression_result.intercept)

    print("mass result:", 1./velocity_regression_result.slope)
    print("translational mu result:", velocity_regression_result.intercept/-9.8)

    print("actual mass value:")
    print("\t", actual_mass)

    print("\n\n\n\nprocessing moment of inertia using known center of mass (assuming we got the other part of it by moving the object along the x-axis)")
    pushing_torques = []
    com_to_first_force = np.array(list(external_force_contact_location_first_force)) - actual_com
    com_to_second_force = np.array(list(external_force_contact_location_second_force)) - actual_com
    for force_magn_pair in force_magn_pairs:
        first_force_magn, second_force_magn = force_magn_pair
        pushing_torques.append(np.cross(com_to_first_force, np.array([dir1[0]*first_force_magn, dir1[1]*first_force_magn, dir1[2]*first_force_magn]))[2] +
                               np.cross(com_to_second_force, np.array([dir2[0]*second_force_magn, dir2[1]*second_force_magn, dir2[2]*second_force_magn]))[2])
    angular_velocity_regression_result = scipy.stats.linregress(pushing_torques, angular_velocity_changes_divided_by_dt)
    print("slope:", angular_velocity_regression_result.slope)
    print("intercept:", angular_velocity_regression_result.intercept)

    print("moment of inertia result:", 1./angular_velocity_regression_result.slope)
    print("intercept should be at the origin.\n")
    #print("actual moment of inertia value:")
    #print("\t",combined.I[1][1])

mass_moments_finder()

p.disconnect()
print("done")