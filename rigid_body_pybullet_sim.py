import pybullet as p
import numpy as np
#import draw_data
import make_URDF
import file_handling
import scipy
import os

import time



#time.sleep(20)

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


def get_actual_mass_com_and_moment_of_inertia():
    #count = 1
    masses = []
    mass = p.getDynamicsInfo(objectID, -1)[0] #base mass
    masses.append(mass)
    #print(count-1,mass,p.getDynamicsInfo(objectID,-1)[1])

    loc_weighed_mass = masses[0]*np.array(list(p.getBasePositionAndOrientation(objectID)[0])) #base location
    num_links = p.getNumJoints(objectID) #excludes base

    for i in range(num_links):
        this_mass = p.getDynamicsInfo(objectID, i)[0]
        masses.append(this_mass)
        #print(count,this_mass,p.getDynamicsInfo(objectID,i)[1])

        this_loc = p.getLinkState(objectID, i)[0]
        mass += this_mass
        loc_weighed_mass += np.array(list(this_loc))*this_mass

        #count += 1
    com = loc_weighed_mass/mass

    I = masses[0]/6. + masses[0]*(np.linalg.norm(np.array(list(p.getBasePositionAndOrientation(objectID)[0])) - com)**2)
    for i in range(num_links):
        I += masses[i+1]/6. + masses[i+1]*(np.linalg.norm(p.getLinkState(objectID, i)[0] - com) ** 2)

    return mass, com, I



def try_a_force_pair(first_force, second_force):
    print("trying a force pair")
    velocity, angular_velocity, friction = run_pybullet_one_time_step(first_force, second_force)

    #record the sample
    push_sample = ""
    first_force_dir_magn, first_force_pos = first_force
    second_force_dir_magn, second_force_pos = second_force
    push_sample += "\n"
    push_sample += str(first_force_pos[0])
    push_sample += "," + str(first_force_pos[1])
    push_sample += "," + str(second_force_pos[0])
    push_sample += "," + str(second_force_pos[1])
    push_sample += "," + str(first_force_dir_magn[0])
    push_sample += "," + str(first_force_dir_magn[1])
    push_sample += "," + str(second_force_dir_magn[0])
    push_sample += "," + str(second_force_dir_magn[1])
    push_sample += "," + str(velocity[0])
    push_sample += "," + str(velocity[1])
    push_sample += "," + str(angular_velocity[2])

    return velocity, angular_velocity[2], friction, push_sample

def find_balancing_external_force_pair(first_force_magn, pos1, dir1, pos2, dir2):
    """Given a force, find the other force such that both forces pushing together cause no rotational motion. This requires trial and error, which will be done here via binary search"""
    push_samples = ""

    force_magn_pairs_found = []
    angular_velocity_changes_found = []
    frictions = []

    #get initial left part of binary search
    second_force_magn = -1. * first_force_magn
    velocity_result, angular_velocity_result, friction, push_sample = try_a_force_pair([(dir1[0]*first_force_magn, dir1[1]*first_force_magn, dir1[2]*first_force_magn), pos1],
                                                                [(dir2[0]*second_force_magn, dir2[1]*second_force_magn, dir2[2]*second_force_magn), pos2])
    push_samples += push_sample
    left = second_force_magn, velocity_result, angular_velocity_result

    #record results for later
    frictions.append(friction)
    force_magn_pairs_found.append((first_force_magn, second_force_magn))
    angular_velocity_changes_found.append(angular_velocity_result)

    #get initial right part of binary search
    second_force_magn = 4. * first_force_magn
    velocity_result, angular_velocity_result, friction, push_sample = try_a_force_pair([(dir1[0]*first_force_magn, dir1[1]*first_force_magn, dir1[2]*first_force_magn), pos1],
                                                                [(dir2[0]*second_force_magn, dir2[1]*second_force_magn, dir2[2]*second_force_magn), pos2])
    push_samples += push_sample
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
        velocity_result, angular_velocity_result, friction, push_sample = try_a_force_pair([(dir1[0]*first_force_magn, dir1[1]*first_force_magn, dir1[2]*first_force_magn), pos1],
                                                                    [(dir2[0]*second_force_magn, dir2[1]*second_force_magn, dir2[2]*second_force_magn), pos2])
        push_samples += push_sample
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

    return result[0], result[1], force_magn_pairs_found, angular_velocity_changes_found, frictions, push_samples


def mass_and_com_one_axis(pos1, dir1, pos2, dir2, first_force_along_axis, second_to_first_force_along_axis, velocity_axis, result_string):
    push_samples = ""

    # magnitudes of the force at each location
    first_force_magns = []
    second_force_magns = []

    # info for getting mass
    forces = []
    delta_velocities_divided_by_dt = []

    # info for getting moment of inertia
    force_magn_pairs = []
    angular_velocity_changes_divided_by_dt = []

    first_force_magn = 7000.
    limit = 2000
    stride = 100

    # friction info. Friction should be constant across all rounds
    frictions = []
    translation_only_motion_frictions = []

    for i in np.arange(0, limit, stride):
        second_force_magn, velocity_result, force_magn_pairs_found, angular_velocity_changes_found, frictions_result, push_samples_result = \
            find_balancing_external_force_pair(first_force_magn, pos1, dir1, pos2, dir2)
        push_samples += push_samples_result

        # add data for translation-only motion
        first_force_magns.append(first_force_magn)
        second_force_magns.append(second_force_magn)
        forces.append(first_force_magn + second_force_magn)
        delta_velocities_divided_by_dt.append(velocity_result[velocity_axis] / dt)

        # add data for motion in general
        translation_only_motion_frictions.append(frictions_result[-1])
        frictions += frictions_result
        force_magn_pairs += force_magn_pairs_found
        for angular_velocity_change in angular_velocity_changes_found:
            angular_velocity_changes_divided_by_dt.append(angular_velocity_change / dt)
        first_force_magn += stride

    # print("force magn pairs:\n", first_force_magns[0], second_force_magns[0], "\n", first_force_magns[1], second_force_magns[1])

    # get com coord along axis
    com_pos_found_along_axis_list = []
    end = len(first_force_magns) - 1
    for i in np.arange(end):
        com_to_first_force_along_axis = second_to_first_force_along_axis * \
                                        (second_force_magns[end] - second_force_magns[i]) / (first_force_magns[end] - first_force_magns[i] + second_force_magns[end] - second_force_magns[i])
        com_pos_found_along_axis = first_force_along_axis - com_to_first_force_along_axis
        com_pos_found_along_axis_list.append(com_pos_found_along_axis)

    '''# draw stuff
    #actual_com_along_axis_list = [actual_com[0]] * len(com_pos_found_along_axis_list)
    #draw_data.plot_data_two_curves(first_force_magns[1:], com_pos_found_along_axis_list, actual_com_along_axis_list)
    #draw_data.plot_data(forces, delta_velocities_divided_by_dt)
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
    draw_data.plot_data(first_force_magns, translation_only_motion_frictions_magn)'''

    # get mass
    print("\n\nprocessing mass")
    result_string += "\nmass result"
    velocity_regression_result = scipy.stats.linregress(forces, delta_velocities_divided_by_dt)
    print("slope:", velocity_regression_result.slope)
    result_string += "\n\tslope: " + str(velocity_regression_result.slope)
    print("intercept:", velocity_regression_result.intercept)
    result_string += "\n\tintercept: " + str(velocity_regression_result.intercept)

    print("mass result:", 1. / velocity_regression_result.slope)
    result_string += "\n\n\tmass result: " + str(1. / velocity_regression_result.slope)
    print("translational mu result:", velocity_regression_result.intercept / -9.8)
    result_string += "\n\ttranslational mu result: " + str(velocity_regression_result.intercept / -9.8)
    found_mass = 1. / velocity_regression_result.slope

    print("actual mass value:")
    result_string += "\nactual mass value: " + str(actual_mass)
    print("\t", actual_mass)
    # return the com for this axis, the force magnitude pairs, and the angular velocities divided by dt
    return found_mass, com_pos_found_along_axis_list[len(com_pos_found_along_axis_list) - 1], force_magn_pairs, angular_velocity_changes_divided_by_dt, result_string, push_samples

def find_positions():
    '''find locations where forces can be exerted'''

    #along x, find min and max x, then minimize y
    min_x = p.getBasePositionAndOrientation(objectID)[0][0]
    max_x = p.getBasePositionAndOrientation(objectID)[0][0]
    num_links = p.getNumJoints(objectID)  # excludes base
    for i in range(num_links):
        x = p.getLinkState(objectID, i)[0][0]
        if x<min_x:
            min_x = x
        if x>max_x:
            max_x = x
    y_for_min_x = None
    y_for_max_x = None
    if min_x == p.getBasePositionAndOrientation(objectID)[0][0]:
        y_for_min_x = p.getBasePositionAndOrientation(objectID)[0][1]
    if max_x == p.getBasePositionAndOrientation(objectID)[0][0]:
        y_for_max_x = p.getBasePositionAndOrientation(objectID)[0][1]
    for i in range(num_links):
        y = p.getLinkState(objectID, i)[0][1]
        if min_x == p.getLinkState(objectID, i)[0][0]:
            if y_for_min_x is None:
                y_for_min_x = y
            elif y<=y_for_min_x:
                y_for_min_x = y
        if max_x == p.getLinkState(objectID, i)[0][0]:
            if y_for_max_x is None:
                y_for_max_x = y
            elif y<=y_for_max_x:
                y_for_max_x = y

    pos_min_x = (min_x, y_for_min_x, 0.5)
    pos_max_x = (max_x, y_for_max_x, 0.5)

    # along y, find max y, then minimize x
    max_y = p.getBasePositionAndOrientation(objectID)[0][1]
    num_links = p.getNumJoints(objectID)  # excludes base
    for i in range(num_links):
        y = p.getLinkState(objectID, i)[0][1]
        if y > max_y:
            max_y = y
    x_for_max_y = None
    if max_y == p.getBasePositionAndOrientation(objectID)[0][1]:
        x_for_max_y = p.getBasePositionAndOrientation(objectID)[0][0]
    for i in range(num_links):
        x = p.getLinkState(objectID, i)[0][0]
        if max_y == p.getLinkState(objectID, i)[0][1]:
            if x_for_max_y is None:
                x_for_max_y = x
            elif x <= x_for_max_y:
                x_for_max_y = x
    pos_max_y = (x_for_max_y, max_y, 0.5)

    return pos_min_x, pos_max_x, pos_max_y

def mass_moments_finder():

    result_string = ""
    push_samples = ""

    #get locations
    pos1, pos2, pos4 = find_positions()
    dir1 = (0., 1., 0.)
    dir2 = (0., 1., 0.)
    external_force_contact_location_first_force = pos1
    first_force_rx = external_force_contact_location_first_force[0]
    
    external_force_contact_location_second_force = pos2
    second_force_rx = external_force_contact_location_second_force[0]
    second_to_first_force_rx = first_force_rx - second_force_rx

    result_string += "pushing along y axis"
    found_mass, com_x, force_magn_pairs, angular_velocity_changes_divided_by_dt, result_string_1, push_samples_result = \
        mass_and_com_one_axis(pos1, dir1, pos2, dir2, first_force_rx, second_to_first_force_rx, 1, result_string)
    result_string = result_string_1
    push_samples += push_samples_result
    result_string += "\n\nnumber of pushes: " + str(len(force_magn_pairs))

    #third force and fourth forces
    third_force_ry = external_force_contact_location_first_force[1]
    dir3 = (1., 0., 0.)
    dir4 = (1., 0., 0.)
    external_force_contact_location_fourth_force = pos4
    fourth_force_ry = external_force_contact_location_fourth_force[1]
    fourth_to_third_force_ry = third_force_ry - fourth_force_ry

    result_string += "\n\n\npushing along z axis"
    found_mass, com_y, force_magn_pairs_unused, angular_velocity_changes_divided_by_dt_unused, result_string_1, push_samples_result = \
        mass_and_com_one_axis(pos1, dir3, pos4, dir4, third_force_ry, fourth_to_third_force_ry, 0, result_string)
    result_string = result_string_1
    push_samples += push_samples_result
    result_string += "\n\nnumber of pushes: " + str(len(force_magn_pairs))

    print("\n\n\n\ncenter of mass result:")
    print("\t", com_x, com_y)
    print("actual value:")
    print("\t", actual_com[0], actual_com[1])

    result_string += "\n\n\n\ncenter of mass result:"
    result_string += "\n\t(" + str(com_x) + "," + str(com_y) + ")"
    result_string += "\nactual center of mass location:"
    result_string += "\n\t(" + str(actual_com[0]) + "," + str(actual_com[1]) + ")"


    print("\n\n\n\nprocessing moment of inertia using the found center of mass")
    result_string += "\n\n\n\nmoment of inertia result using the found center of mass"
    pushing_torques = []
    found_com = np.array([com_x, com_y, external_force_contact_location_first_force[2]])
    com_to_first_force = np.array(list(external_force_contact_location_first_force)) - found_com
    com_to_second_force = np.array(list(external_force_contact_location_second_force)) - found_com
    for force_magn_pair in force_magn_pairs:
        first_force_magn, second_force_magn = force_magn_pair
        pushing_torques.append(np.cross(com_to_first_force, np.array([dir1[0]*first_force_magn, dir1[1]*first_force_magn, dir1[2]*first_force_magn]))[2] +
                               np.cross(com_to_second_force, np.array([dir2[0]*second_force_magn, dir2[1]*second_force_magn, dir2[2]*second_force_magn]))[2])
    angular_velocity_regression_result = scipy.stats.linregress(pushing_torques, angular_velocity_changes_divided_by_dt)
    print("slope:", angular_velocity_regression_result.slope)
    result_string += "\n\tslope: " + str(angular_velocity_regression_result.slope)
    print("intercept:", angular_velocity_regression_result.intercept)
    result_string += "\n\tintercept: " + str(angular_velocity_regression_result.intercept)

    print("moment of inertia result:", 1./angular_velocity_regression_result.slope)
    result_string += "\n\n\tmoment of inertia result: " + str(1./angular_velocity_regression_result.slope)
    print("intercept should be at the origin.\n")
    result_string += "\n\tintercept should be at the origin."
    print("actual moment of inertia value:")
    print("\t",actual_I)
    result_string += "\n\nactual moment of inertia value: " + str(actual_I)
    found_I = 1./angular_velocity_regression_result.slope


    print("\n\n\n\nprocessing moment of inertia using the actual center of mass")
    result_string += "\n\n\n\nmoment of inertia result using the actual center of mass"
    pushing_torques = []
    com_to_first_force = np.array(list(external_force_contact_location_first_force)) - actual_com
    com_to_second_force = np.array(list(external_force_contact_location_second_force)) - actual_com
    for force_magn_pair in force_magn_pairs:
        first_force_magn, second_force_magn = force_magn_pair
        pushing_torques.append(np.cross(com_to_first_force, np.array([dir1[0]*first_force_magn, dir1[1]*first_force_magn, dir1[2]*first_force_magn]))[2] +
                               np.cross(com_to_second_force, np.array([dir2[0]*second_force_magn, dir2[1]*second_force_magn, dir2[2]*second_force_magn]))[2])
    angular_velocity_regression_result = scipy.stats.linregress(pushing_torques, angular_velocity_changes_divided_by_dt)
    print("slope:", angular_velocity_regression_result.slope)
    result_string += "\n\tslope: " + str(angular_velocity_regression_result.slope)
    print("intercept:", angular_velocity_regression_result.intercept)
    result_string += "\n\tintercept: " + str(angular_velocity_regression_result.intercept)

    print("moment of inertia result:", 1./angular_velocity_regression_result.slope)
    result_string += "\n\n\tmoment of inertia result: " + str(1./angular_velocity_regression_result.slope)
    print("intercept should be at the origin.\n")
    result_string += "\n\tintercept should be at the origin."
    print("actual moment of inertia value:")
    print("\t",actual_I)
    result_string += "\n\nactual moment of inertia value: " + str(actual_I)

    result_string += "\n"
    return result_string, push_samples, found_mass, found_com, found_I




def write_results(result_string, push_samples, dir_name, testNum):
    # set up data storage
    result_file = os.path.join(dir_name, "test " + str(testNum) + " results.txt")
    push_samples_file = os.path.join(dir_name, "push_samples.csv")
    result_records = open(result_file, "w")
    push_samples_records = open(push_samples_file, "w")
    result_records.write(result_string)

    push_samples_records.write("force1" + "_x_loc")
    push_samples_records.write(",force1" + "_y_loc")
    push_samples_records.write(",force2" + "_x_loc")
    push_samples_records.write(",force2" + "_y_Loc")
    push_samples_records.write(",force1" + "_x_dir")
    push_samples_records.write(",force1" + "_y_dir")
    push_samples_records.write(",force2" + "_x_dir")
    push_samples_records.write(",force2" + "_y_dir")
    push_samples_records.write(",post-push object" + "_velocity_x")
    push_samples_records.write(",post-push object" + "_velocity_y")
    push_samples_records.write(",post-push object" + "_angular_velocity_z")
    push_samples_records.write(push_samples)

    result_records.close()
    push_samples_records.close()




actual_mass = actual_com = actual_I = None
objectID = None
dt = 1./240.
startPos = [0, 0, 0.5]
startOrientation = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
def run_full_test(object_name):
    # make directory for simulation files
    test_num = 1
    while os.path.exists("test" + str(test_num)):
        test_num += 1
    test_dir = "test" + str(test_num)
    os.mkdir(test_dir)

    #make the file being tested
    make_URDF.write_URDF(object_name, test_dir)

    physicsClient = p.connect(p.DIRECT)
    #physicsClient = p.connect(p.GUI)
    p.setGravity(0,0,-9.8)

    global objectID
    objectID = p.loadURDF(test_dir + "\\" + object_name + ".urdf")
    p.resetBasePositionAndOrientation(objectID, startPos, startOrientation)

    planeID = p.loadURDF("urdf and meshes\plane.urdf")
    print("object:",objectID)
    print("plane:",planeID)

    global actual_mass
    global actual_com
    global actual_I
    actual_mass, actual_com, actual_I = get_actual_mass_com_and_moment_of_inertia() #global values
    result_string, push_samples, found_mass, found_com, found_I = mass_moments_finder()

    print("Finished with simulation. Writing result files.")

    write_results(result_string, push_samples, test_dir, test_num)

    print("Finished writing result files. Writing visualization files.")

    shapes_loc_data = [np.array(list(p.getBasePositionAndOrientation(objectID)[0]))]
    masses_data = [p.getDynamicsInfo(objectID, -1)[0]]
    frictions_data = [p.getDynamicsInfo(objectID, -1)[1]]
    num_links = p.getNumJoints(objectID)  # excludes base
    for i in range(num_links):
        shapes_loc_data.append(np.array(list(p.getLinkState(objectID, i)[0])))
        masses_data.append(p.getDynamicsInfo(objectID, i)[0])
        frictions_data.append(p.getDynamicsInfo(objectID,i)[1])

    file_handling.write_obj_and_mtl_one_frame(shapes_loc_data, masses_data, frictions_data, actual_com, found_com, object_name, test_dir, "mass")
    file_handling.write_obj_and_mtl_one_frame(shapes_loc_data, masses_data, frictions_data, actual_com, found_com, object_name, test_dir, "friction")

    p.disconnect()

    print("Done.")


run_full_test("spray_gun_uniform")
run_full_test("spray_gun")
run_full_test("snack_uniform")
run_full_test("snack")
run_full_test("book_uniform")
run_full_test("book")
run_full_test("wrench_uniform")
run_full_test("wrench")
run_full_test("hammer_uniform")
run_full_test("hammer")
