import os
import numpy as np
import geometry_utils

#file handling
def read_obj_mesh(file_loc):
    file = open(file_loc, "r", encoding="utf-8")
    stuff = file.readlines()
    file.close()

    vertices = []
    normals = []
    faces = []

    #handle vertices
    center_to_subtract = np.array([0., 0., 0.])
    for line in stuff:
        line_stripped = line.strip()
        if line_stripped.startswith("v "):
            vec = line_stripped[2:].split(" ")
            for i in np.arange(len(vec)):
                vec[i] = float(vec[i])
            vertex_to_add = np.array(vec)
            vertices.append(vertex_to_add)
            center_to_subtract += vertex_to_add
    center_to_subtract /= len(vertices)
    for vertex in vertices:
        vertex -= center_to_subtract

    #handle normals
    for line in stuff:
        line_stripped = line.strip()
        if line_stripped.startswith("vn "):
            vec = line_stripped[3:].split(" ")
            for i in np.arange(len(vec)):
                vec[i] = float(vec[i])
            normal_to_add = np.array(vec)
            normals.append(normal_to_add)

    #handle faces
    for line in stuff:
        line_stripped = line.strip()
        if line_stripped.startswith("f "):
            vec = line_stripped[2:].split(" ")
            face_vertices_indices = []
            normal_index = None
            for i in np.arange(len(vec)):
                vec[i] = vec[i].split("/")
                face_vertices_indices.append(int(vec[i][0]) - 1)
                normal_index = int(vec[i][2]) - 1 #normal should be the same for every vertex in the face
            faces.append((face_vertices_indices, normal_index))

    return vertices, normals, faces


def read_motion_script_file(motion_script_file_to_use):
    motion_script_data_file = open(motion_script_file_to_use, "r")
    motion_script_raw = motion_script_data_file.read()
    motion_script_data_file.close()
    motion_script_raw_lines = motion_script_raw.split("\n")[1:-1]
    motion_script = []

    for line in motion_script_raw_lines:
        split_line = line.split(",")
        motion_script_line = []

        for split_line_part in split_line[:14]:
            motion_script_line.append(float(split_line_part))

        if len(split_line) > 14:
            #add contacts for the current time step
            i = 14
            while i < len(split_line):
                type_of_contact = split_line[i]
                motion_script_line.append(type_of_contact)
                if type_of_contact == "shape-shape_contact":
                    motion_script_line.append(int(split_line[i + 1]))
                    motion_script_line.append(int(split_line[i + 2]))
                    motion_script_line.append(float(split_line[i + 3]))
                    motion_script_line.append(float(split_line[i + 4]))
                    motion_script_line.append(float(split_line[i + 5]))
                    motion_script_line.append(float(split_line[i + 6]))
                    motion_script_line.append(float(split_line[i + 7]))
                    motion_script_line.append(float(split_line[i + 8]))
                    i = i + 9
                elif type_of_contact == "ground-shape_contact":
                    motion_script_line.append(int(split_line[i + 1]))
                    motion_script_line.append(float(split_line[i + 2]))
                    motion_script_line.append(float(split_line[i + 3]))
                    motion_script_line.append(float(split_line[i + 4]))
                    motion_script_line.append(float(split_line[i + 5]))
                    motion_script_line.append(float(split_line[i + 6]))
                    motion_script_line.append(float(split_line[i + 7]))
                    i = i + 8
                elif type_of_contact == "external_force":
                    motion_script_line.append(float(split_line[i + 1]))
                    motion_script_line.append(float(split_line[i + 2]))
                    motion_script_line.append(float(split_line[i + 3]))
                    motion_script_line.append(float(split_line[i + 4]))
                    motion_script_line.append(float(split_line[i + 5]))
                    i = i + 6
                else:
                    print("error: cannot read contact type in existing motion script: ", type_of_contact)
                    exit()

        motion_script.append(motion_script_line)
    return motion_script

def read_external_force_script_file(external_force_script_file_to_use):
    external_force_script_data_file = open(external_force_script_file_to_use, "r")
    external_force_script_raw = external_force_script_data_file.read()
    external_force_script_data_file.close()
    external_force_script_raw_lines = external_force_script_raw.split("\n")[1:-1]
    external_force_script = []

    for line in external_force_script_raw_lines:
        split_line = line.split(",")
        external_force_script_line = []

        external_force_script_line.append(float(split_line[0]))
        external_force_script_line.append(int(split_line[1]))
        external_force_script_line.append(int(split_line[2]))

        external_force_script.append(external_force_script_line)
    return external_force_script

def setup_records_and_motion_script(shapes):
    # make directory for simulation files
    testNum = 1
    while os.path.exists("test" + str(testNum)):
        testNum += 1
    dir_name = "test" + str(testNum)
    os.mkdir(dir_name)

    # set up data storage
    loc_file = os.path.join(dir_name, "data_locations.csv")
    motion_script_loc = os.path.join(dir_name, "motion_script.csv")
    locations_records = open(loc_file, "w")
    motion_script = open(motion_script_loc, "w")
    locations_records.write("time")
    motion_script.write("time")
    motion_script.write(",combined" + "_x")
    motion_script.write(",combined" + "_y")
    motion_script.write(",combined" + "_z")
    motion_script.write(",combined" + "_quaternion_i")
    motion_script.write(",combined" + "_quaternion_j")
    motion_script.write(",combined" + "_quaternion_k")
    motion_script.write(",combined" + "_quaternion_s")
    motion_script.write(",combined" + "_velocity_x")
    motion_script.write(",combined" + "_velocity_y")
    motion_script.write(",combined" + "_velocity_z")
    motion_script.write(",combined" + "_angular_velocity_x")
    motion_script.write(",combined" + "_angular_velocity_y")
    motion_script.write(",combined" + "_angular_velocity_z")
    for count in np.arange(len(shapes)):
        locations_records.write(",shape_" + str(count) + "_x")
        locations_records.write(",shape_" + str(count) + "_y")
        locations_records.write(",shape_" + str(count) + "_z")
        locations_records.write(",shape_" + str(count) + "_quaternion_i")
        locations_records.write(",shape_" + str(count) + "_quaternion_j")
        locations_records.write(",shape_" + str(count) + "_quaternion_k")
        locations_records.write(",shape_" + str(count) + "_quaternion_s")
    locations_records.write("\n")
    motion_script.write(",contacts if any (8 columns per ground-shape contact and 9 columns per shape-shape contact) plus 6 columns at the end for the external force\n")

    energies_file = os.path.join(dir_name, "data_energy.csv")
    energies_records = open(energies_file, "w")
    energies_records.write("time,KE,PE,total energy\n")
    # momenta_file = os.path.join(dir_name, "data_momenta.csv")
    # angular momentum file

    return locations_records, energies_records, motion_script, loc_file, dir_name

def write_records_and_motion_script(locations_records, motion_script, energies_records, time, total_KE, total_PE, total_energy, combined, shapes):
    locations_records.write(str(time))
    motion_script.write(str(time))
    energies_records.write(str(time))
    energies_records.write("," + str(total_KE) + "," + str(total_PE) + "," + str(total_energy) + "\n")
    for coord in combined.location:
        motion_script.write("," + str(coord))
    for coord in combined.orientation:
        motion_script.write("," + str(coord))
    for coord in combined.velocity:
        motion_script.write("," + str(coord))
    for coord in combined.angular_velocity:
        motion_script.write("," + str(coord))
    for shape in shapes:
        for coord in shape.location:
            locations_records.write("," + str(coord))
        for coord in shape.orientation:
            locations_records.write("," + str(coord))
    locations_records.write("\n")

def write_contacts_in_motion_script(shapes, shape_shape_contacts_low_level, ground_contacts_low_level, motion_script):
    for contact in shape_shape_contacts_low_level:
        shape_pair, contact_info = contact
        shape1, shape2 = shape_pair
        contact_location, normal = contact_info
        motion_script.write(",shape-shape_contact," + str(shapes.index(shape1)) + "," + str(shapes.index(shape2)))
        motion_script.write("," + str(contact_location[0]))
        motion_script.write("," + str(contact_location[1]))
        motion_script.write("," + str(contact_location[2]))
        motion_script.write("," + str(normal[0]))
        motion_script.write("," + str(normal[1]))
        motion_script.write("," + str(normal[2]))
    for contact in ground_contacts_low_level:
        shape, contact_info = contact
        contact_location, normal = contact_info
        motion_script.write(",ground-shape_contact," + str(shapes.index(shape)))
        motion_script.write("," + str(contact_location[0]))
        motion_script.write("," + str(contact_location[1]))
        motion_script.write("," + str(contact_location[2]))
        motion_script.write("," + str(normal[0]))
        motion_script.write("," + str(normal[1]))
        motion_script.write("," + str(normal[2]))

def write_external_force_info_in_motion_script(external_force_magn, external_force_contact_location, direction_z, motion_script):
    motion_script.write(",external_force")
    motion_script.write(","+str(external_force_magn))
    motion_script.write(","+str(external_force_contact_location[0]))
    motion_script.write(","+str(external_force_contact_location[1]))
    motion_script.write(","+str(external_force_contact_location[2]))
    motion_script.write(","+str(direction_z))
    motion_script.write("\n")


def write_frames(shapes, locations_data_file, dir_name, dt, fps, mode):
    # get data on component shape locations
    locations_data = open(locations_data_file, "r", encoding="utf-8")
    data = locations_data.read()
    locations_data.close()

    if mode is not None:
        #assume for now that max mass is 20 and max friction is 0.5
        max_mass = 20.
        max_friction = .5

        # get masses and frictions data
        mass_and_friction_data_file = os.path.join(os.path.dirname(dir_name), "masses_and_frictions.csv")
        mass_and_friction_data = open(mass_and_friction_data_file, "r", encoding="utf-8")
        masses = []
        frictions = []
        lines = mass_and_friction_data.read().split("\n")
        for line in lines[1:-1]:
            data_row = line.split(",")
            masses.append(float(data_row[0]))
            frictions.append(float(data_row[1]))
        mass_and_friction_data.close()

        #write material file
        material = open(os.path.join(dir_name, "materials.mtl"), "w")
        if mode == "mass":
            for shape_count in range(len(shapes)):
                # mass material, green = maximum mass, red = massless
                material.write("newmtl " + "{:0>4d}".format(shape_count) + "\n")
                amount = masses[shape_count] / max_mass
                material.write("Ka " + str(1. - amount) + " " + str(amount) + " 0.0\n")
                material.write("Kd " + str(1. - amount) + " " + str(amount) + " 0.0\n")
                material.write("Ks 0.0 0.0 0.0\n")
        elif mode == "friction":
            for shape_count in range(len(shapes)):
                # friction material, green = maximum friction, red = frictionless
                material.write("newmtl " + "{:0>4d}".format(shape_count) + "\n")
                amount = frictions[shape_count] / max_friction
                material.write("Ka " + str(1. - amount) + " " + str(amount) + " 0.0\n")
                material.write("Kd " + str(1. - amount) + " " + str(amount) + " 0.0\n")
                material.write("Ks 0.0 0.0 0.0\n")
        elif mode == "mass_times_friction":
            for shape_count in range(len(shapes)):
                # mass*friction material, green = maximum mass*friction, red = massless and/or frictionless
                material.write("newmtl " + "{:0>4d}".format(shape_count) + "\n")
                amount = masses[shape_count] * frictions[shape_count] / (max_mass * max_friction)
                material.write("Ka " + str(1. - amount) + " " + str(amount) + " 0.0\n")
                material.write("Kd " + str(1. - amount) + " " + str(amount) + " 0.0\n")
                material.write("Ks 0.0 0.0 0.0\n")
        material.close()

    # write fromes for blender
    lines = data.split("\n")
    frame_count = 1
    line_count = 0
    skip = round(1. / (dt * fps))
    for line in lines[1:-1]:
        if (line_count % skip != 0):
            line_count += 1
            continue
        data_row = line.split(",")
        for i in np.arange(len(data_row)):
            data_row[i] = float(data_row[i])
        out_file = open(os.path.join(dir_name, "frame_" + "{:0>4d}".format(frame_count) + ".obj"), "w")
        time = data_row[0]
        out_file.write("#" + str(time) + "\n")
        if mode is not None:
            out_file.write("mtllib materials.mtl\n")  # materials file to be generated by blender
        vertices_index_shift = [0]
        for shape_count, shape in enumerate(shapes):
            shape_data = data_row[1 + 7 * shape_count:1 + 7 * (shape_count + 1)]
            loc = np.array(shape_data[:3])
            orientation = np.array(shape_data[3:])

            # write vertices
            vertices_added = 0
            for vertex in shape.vertices:
                world_vertex = geometry_utils.rotation_from_quaternion(orientation,vertex) + loc  # same as geometry_utils.to_world_coords but shape is not available since its past data is only in the file
                out_file.write("v")
                for coord in world_vertex:
                    out_file.write(" " + str(coord))
                out_file.write("\n")
                vertices_added += 1
            vertices_index_shift.append(vertices_added)

        # update vertex indicies for writing faces
        for i in np.arange(1, len(vertices_index_shift)):
            vertices_index_shift[i] += vertices_index_shift[i - 1]

        for shape_count, shape in enumerate(shapes):
            if mode is not None:
                out_file.write("usemtl " + "{:0>4d}".format(shape_count) + "\n")
            # write faces
            for vertex_indices, normal_index in shape.faces:
                out_file.write("f")
                for vertex_index in vertex_indices:
                    out_file.write(" " + str(vertex_index + vertices_index_shift[shape_count] + 1))
                out_file.write("\n")
        frame_count += 1
        line_count += 1

def write_simulation_files(shapes, shape_ground_frictions, locations_data_file, dir_name, dt, fps):
    #print out to a file the component shapes' masses and frictions
    masses_and_frictions = open(os.path.join(dir_name,"masses_and_frictions.csv"), "w")
    masses_and_frictions.write("mass,friction,mass*friction\n")
    for shape in shapes:
        masses_and_frictions.write(str(shape.mass)+","+str(shape_ground_frictions[shape])+","+str(shape.mass*shape_ground_frictions[shape])+"\n")
    masses_and_frictions.close()


    plain_frames_dir = os.path.join(dir_name, "frames")
    os.mkdir(plain_frames_dir)
    write_frames(shapes, locations_data_file, plain_frames_dir, dt, fps, None)

    mass_frames_dir = os.path.join(dir_name, "mass_frames")
    os.mkdir(mass_frames_dir)
    write_frames(shapes, locations_data_file, mass_frames_dir, dt, fps, "mass")

    friction_frames_dir = os.path.join(dir_name, "friction_frames")
    os.mkdir(friction_frames_dir)
    write_frames(shapes, locations_data_file, friction_frames_dir, dt, fps, "friction")

    mass_times_friction_frames_dir = os.path.join(dir_name, "mass_times_friction_frames")
    os.mkdir(mass_times_friction_frames_dir)
    write_frames(shapes, locations_data_file, mass_times_friction_frames_dir, dt, fps, "mass_times_friction")











def read_combined_boxes_rigid_body_file(file_loc):
    file = open(file_loc, "r", encoding="utf-8")
    stuff = file.readlines()
    file.close()

    info = []
    for line in stuff:
        line_stripped = line.strip()
        x,y,z = line_stripped.split(",")
        info.append((float(x), float(y), float(z)))
    return info

def read_combined_boxes_extra_info_rigid_body_file(file_loc):
    file = open(file_loc, "r", encoding="utf-8")
    stuff = file.readlines()
    file.close()

    info = []
    for line in stuff[1:]:
        line_stripped = line.strip()
        x,y,z,mass,friction = line_stripped.split(",")
        info.append((float(x), float(y), float(z), float(mass), float(friction)))
    return info

def write_urdf(object_loc, object_name, urdf_str):
    file = open(os.path.join(object_loc, object_name) +".urdf", "w", encoding="utf-8")
    file.write(urdf_str)
    file.close()

def write_obj_and_mtl_one_frame(shapes_loc_data, masses_data, frictions_data, com_actual_loc, com_found_loc, shape_name, dir_name, mode):

    #assume for now that max mass is 20 and max friction is 1.
    max_mass = 20.
    max_friction = 1.

    #write material file
    material = open(os.path.join(dir_name, "materials_"+mode+".mtl"), "w")
    material.write("newmtl com_actual\n") # material for actual center of mass
    material.write("Ka 1. 1. 1.\n")
    material.write("Kd 1. 1. 1.\n")
    material.write("Ks 0.0 0.0 0.0\n")
    material.write("newmtl com_found\n") # material for found center of mass
    material.write("Ka 0. 0. 0.\n")
    material.write("Kd 0. 0. 0.\n")
    material.write("Ks 0.0 0.0 0.0\n")
    if mode == "mass":
        for shape_count in range(len(shapes_loc_data)):
            # mass material, green = maximum mass, red = massless
            material.write("newmtl " + "{:0>4d}".format(shape_count) + "\n")
            amount = masses_data[shape_count] / max_mass
            material.write("Ka " + str(1.-amount) + " " + str(amount) + " " + " 0.\n")
            material.write("Kd " + str(1.-amount) + " " + str(amount) + " " + " 0.\n")
            material.write("Ks 0.0 0.0 0.0\n")
    elif mode == "friction":
        for shape_count in range(len(shapes_loc_data)):
            # friction material, green = maximum friction, red = frictionless
            material.write("newmtl " + "{:0>4d}".format(shape_count) + "\n")
            amount = frictions_data[shape_count] / max_friction
            material.write("Ka " + str(1.-amount) + " " + str(amount) + " " + " 0.\n")
            material.write("Kd " + str(1.-amount) + " " + str(amount) + " " + " 0.\n")
            material.write("Ks 0.0 0.0 0.0\n")
    elif mode == "mass_times_friction":
        for shape_count in range(len(shapes_loc_data)):
            # mass*friction material, green = maximum mass*friction, red = massless and/or frictionless
            material.write("newmtl " + "{:0>4d}".format(shape_count) + "\n")
            amount = masses_data[shape_count] * frictions_data[shape_count] / (max_mass * max_friction)
            material.write("Ka " + str(1.-amount) + " " + str(amount) + " " + " 0.\n")
            material.write("Kd " + str(1.-amount) + " " + str(amount) + " " + " 0.\n")
            material.write("Ks 0.0 0.0 0.0\n")
    material.close()

    # write a frame for blender
    out_file = open(os.path.join(dir_name, shape_name + "_" + mode + ".obj"), "w")
    out_file.write("#" + shape_name + "\n")
    out_file.write("mtllib materials_" + mode + ".mtl\n")  # materials file to be generated by blender

    #write vertices for actual and found center of mass
    point_vertices = [np.array([-0.1, 0., 1.01]), np.array([0., -0.1, 1.01]), np.array([0.1, 0., 1.01]), np.array([0., 0.1, 1.01])]
    for vertex in point_vertices:
        world_vertex = vertex + com_actual_loc
        out_file.write("v")
        for coord in world_vertex:
            out_file.write(" " + str(coord))
        out_file.write("\n")
    for vertex in point_vertices:
        world_vertex = vertex + com_found_loc
        out_file.write("v")
        for coord in world_vertex:
            out_file.write(" " + str(coord))
        out_file.write("\n")

    # write vertices for component shapes
    vertices_index_shift = [8]
    vertices = [np.array([-0.5, -0.5, 0.]), np.array([-0.5, 0.5, 0.]), np.array([0.5, -0.5, 0.]), np.array([0.5, 0.5, 0.]),
                np.array([-0.5, -0.5, 1.]), np.array([-0.5, 0.5, 1.]), np.array([0.5, -0.5, 1.]), np.array([0.5, 0.5, 1.])]
    for shape_loc_data in shapes_loc_data:
        loc = np.array(shape_loc_data)
        vertices_added = 0
        for vertex in vertices:
            world_vertex = vertex + loc
            out_file.write("v")
            for coord in world_vertex:
                out_file.write(" " + str(coord))
            out_file.write("\n")
            vertices_added += 1
        vertices_index_shift.append(vertices_added)

    # update vertex indicies for writing faces
    for i in np.arange(1, len(vertices_index_shift)):
        vertices_index_shift[i] += vertices_index_shift[i - 1]

    # write faces for actual and found centers of mass
    out_file.write("usemtl com_actual\n")
    out_file.write("f")
    for i in np.arange(4):
        out_file.write(" " + str(i + 1))
    out_file.write("\n")
    out_file.write("usemtl com_found\n")
    out_file.write("f")
    for i in np.arange(4):
        out_file.write(" " + str(i + 5))
    out_file.write("\n")

    # write faces for component shapes
    faces = [([0, 1, 3, 2], 0),
             ([4, 5, 7, 6], 1),
             ([0, 1, 5, 4], 2),
             ([2, 3, 7, 6], 3),
             ([0, 2, 6, 4], 4),
             ([1, 3, 7, 5], 5)]
    for shape_count in np.arange(len(shapes_loc_data)):
        out_file.write("usemtl " + "{:0>4d}".format(shape_count) + "\n")

        for vertex_indices, normal_index in faces:
            out_file.write("f")
            for vertex_index in vertex_indices:
                out_file.write(" " + str(vertex_index + vertices_index_shift[shape_count] + 1))
            out_file.write("\n")
