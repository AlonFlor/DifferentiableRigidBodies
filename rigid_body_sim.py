import numpy as np
import scipy
import os
import collision_handling_basic_impulse
import collision_detection
import geometry_utils
import file_handling
import draw_data


#rigid body classes
class collision_shape:
    def __init__(self, shape_type, dimensions, location, orientation, velocity, angular_velocity, mass):
        self.shape_type = shape_type
        self.parent = None

        #add mesh-related data
        if self.shape_type == "box":
            self.x_len, self.y_len, self.z_len = dimensions
            vertices, face_normals, faces = geometry_utils.generate_box(dimensions)
            self.vertices = vertices
            self.face_normals = face_normals
            self.faces = faces                  #each face is a tuple (vertex_indices,normal_index). These indices point to vertices in self.vertices and a normal in self.face_normals
            edges = {}
            for vertex_indices,normal_index in self.faces:
                for i in np.arange(-1,len(vertex_indices)-1):
                    edge = (vertex_indices[i],vertex_indices[i+1])
                    already_taken = False
                    if edge in edges:
                        already_taken = True
                    if (not already_taken) and ((edge[1],edge[0]) in edges):
                        already_taken = True
                    if not already_taken:
                        edges[edge] = 1
            self.edges = []                 #each edge is a tuple of two vertex indices, pointing to vertices from self.vertices
            for edge in edges:
                self.edges.append(edge)
        elif self.shape_type == "sphere":
            self.radius = dimensions
            #vertices, face normals, and faces are used only to print out the shape at the end of the simulation
            vertices, face_normals, faces = geometry_utils.generate_sphere(dimensions)
            self.vertices = vertices
            self.face_normals = face_normals
            self.faces = faces
        elif self.shape_type == "uncapped cylinder":
            self.radius, self.z_len = dimensions
            self.end1 = np.array([0.,0.,-1*self.z_len/2])
            self.end2 = np.array([0.,0.,self.z_len/2])
            #vertices, face normals, and faces are used only to print out the shape at the end of the simulation
            vertices, face_normals, faces = geometry_utils.generate_uncapped_cylinder(dimensions)
            self.vertices = vertices
            self.face_normals = face_normals
            self.faces = faces

        else:
            print("error - unsupported shape type")
            exit()

        #add location-related data
        self.location = location
        self.orientation = orientation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.mass = mass
        self.center = np.array([0., 0., 0.])
        #add moment of inertia and center of mass
        self.I = np.zeros((3,3))
        self.COM = self.center
        if self.shape_type == "box":
            #uniform cuboid
            self.I[0][0] = self.mass*(self.y_len*self.y_len + self.z_len*self.z_len)/12.
            self.I[1][1] = self.mass*(self.x_len*self.x_len + self.z_len*self.z_len)/12.
            self.I[2][2] = self.mass*(self.x_len*self.x_len + self.y_len*self.y_len)/12.
        elif self.shape_type == "sphere":
            #uniform sphere
            self.I[0][0] = 0.4*self.mass*self.radius*self.radius
            self.I[1][1] = self.I[0][0]
            self.I[2][2] = self.I[0][0]
        elif self.shape_type == "uncapped cylinder":
            #uniform cylinder
            self.I[0][0] = self.mass*(3*self.radius*self.radius + self.z_len*self.z_len)
            self.I[1][1] = self.I[0][0]
            self.I[2][2] = 0.5*self.mass*self.radius*self.radius
        self.I_inv = np.linalg.inv(self.I)
        #initialize orientation derivative
        self.orientation_derivative = geometry_utils.orientation_derivative(self.orientation, self.angular_velocity)
        #initialize energy
        self.KE = 0.5*self.mass*np.square(np.linalg.norm(self.velocity)) + 0.5*np.dot(np.matmul(self.I,self.angular_velocity), self.angular_velocity)
        self.PE = 0
        #initialize bounding box
        self.bounding_box = None #will get updated during loop

        #set up signed distance function
        if self.shape_type == "box":
            self.faces_sdf = []
            for face in self.faces:
                self.faces_sdf.append(collision_detection.generate_signed_distance_function_for_face(self.center, self.vertices, self.face_normals, face))
            self.signed_distance = collision_detection.generate_signed_distance_function_for_faces(self.faces_sdf)

        elif self.shape_type == "sphere":
            self.signed_distance = collision_detection.generate_signed_distance_function_sphere(self.radius)
        elif self.shape_type == "uncapped cylinder":
            self.signed_distance = collision_detection.generate_signed_distance_function_cylinder(self.radius)


        #set up mass derivatives of I and I_inv
        self.I_mass_derivative = np.zeros((3,3))
        if self.shape_type == "box":
            #uniform cuboid
            self.I_mass_derivative[0][0] = (self.y_len*self.y_len + self.z_len*self.z_len)/12.
            self.I_mass_derivative[1][1] = (self.x_len*self.x_len + self.z_len*self.z_len)/12.
            self.I_mass_derivative[2][2] = (self.x_len*self.x_len + self.y_len*self.y_len)/12.
        elif self.shape_type == "sphere":
            #uniform sphere
            self.I_mass_derivative[0][0] = 0.4*self.radius*self.radius
            self.I_mass_derivative[1][1] = self.I_mass_derivative[0][0]
            self.I_mass_derivative[2][2] = self.I_mass_derivative[0][0]
        elif self.shape_type == "uncapped cylinder":
            #uniform cylinder
            self.I_mass_derivative[0][0] = (3*self.radius*self.radius + self.z_len*self.z_len)
            self.I_mass_derivative[1][1] = self.I_mass_derivative[0][0]
            self.I_mass_derivative[2][2] = 0.5*self.radius*self.radius
        self.I_inv_mass_derivatives = [-1*self.I_inv*self.I_inv*self.I_mass_derivative] #multiply component by component

class combined_body:
    def __init__(self, components, velocity, angular_velocity):
        self.components = components
        for shape in components:
            shape.parent = self

        ##add location-related data
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.orientation_derivative = geometry_utils.orientation_derivative(self.orientation, self.angular_velocity)

        #add mass and center of mass information
        self.mass= 0.
        world_COM = np.array([0., 0., 0.])
        self.location = np.array([0., 0., 0.])
        for shape in self.components:
            self.location += shape.location
            self.mass += shape.mass
            world_COM += shape.mass*geometry_utils.to_world_coords(shape, shape.COM)
        self.location /= len(self.components)   #location is non-weighted average of component locations
        self.COM = world_COM / self.mass - self.location

        #component rotations from this shape to child shape
        #component translations from this shape to child shape
        self.component_rotations = []
        self.component_translations = []
        for shape in self.components:
            self.component_rotations.append(shape.orientation + 0.)
            self.component_translations.append(shape.location - self.location)

        #get moment of inertia and its inverse
        self.I = np.zeros((3,3))
        for index in np.arange(len(self.components)):
            shape = self.components[index]
            displacement = self.component_translations[index]
            translated_shape_I = shape.I + shape.mass*(np.dot(displacement, displacement)*np.identity(3) - np.outer(displacement, displacement))
            R = geometry_utils.quaternion_to_rotation_matrix(self.component_rotations[index])
            self.I += np.matmul(R, np.matmul(translated_shape_I, R.T))
        self.I_inv = np.linalg.inv(self.I)

        #get mass derivatives of combined body's center of mass
        self.COM_mass_derivatives = []
        second_term = world_COM / (self.mass * self.mass)
        for index in np.arange(len(self.components)):
            shape = self.components[index]
            self.COM_mass_derivatives.append(geometry_utils.to_world_coords(shape, shape.COM) / self.mass - second_term)
            #d_COM/d_mass = (d_world_COM/d_mass) / self.mass - world_COM / (self.mass * self.mass)
        #get mass derivatives of moment of inertia and its inverse, and of its location
        self.I_mass_derivatives = []
        for index in np.arange(len(self.components)):
            self.I_mass_derivatives.append(np.zeros((3,3)))
            shape = self.components[index]
            displacement = self.component_translations[index]
            translated_shape_I_mass_derivative = shape.I_mass_derivative + (np.dot(displacement, displacement)*np.identity(3) - np.outer(displacement, displacement))
            R = geometry_utils.quaternion_to_rotation_matrix(self.component_rotations[index])
            self.I_mass_derivatives[index] += np.matmul(R, np.matmul(translated_shape_I_mass_derivative, R.T))
        self.I_inv_mass_derivatives = []
        for index in np.arange(len(self.components)):
            self.I_inv_mass_derivatives.append(-1*np.matmul(self.I_inv, np.matmul(self.I_mass_derivatives[index], self.I_inv)))

        #set the indices of component shapes as not interacting with each other
        self.fixed_contacts = []
        for i in np.arange(len(self.components)):
            for j in np.arange(i + 1, len(self.components)):
                self.fixed_contacts.append((self.components[i], self.components[j]))

        #set empty velocity derivatives and angular velocity derivatives with respect to mass and mu.
        self.velocity_mass_derivatives = []
        self.angular_velocity_mass_derivatives = []
        self.velocity_mu_derivatives = []
        self.angular_velocity_mu_derivatives = []
        for component in self.components:
            self.velocity_mass_derivatives.append(np.array([0., 0., 0.]))
            self.angular_velocity_mass_derivatives.append(np.array([0., 0., 0.]))
            self.velocity_mu_derivatives.append(np.array([0., 0., 0.]))
            self.angular_velocity_mu_derivatives.append(np.array([0., 0., 0.]))

    def reset_velocity_and_angular_velocity_derivatives(self):
        self.velocity_mass_derivatives = []
        self.angular_velocity_mass_derivatives = []
        self.velocity_mu_derivatives = []
        self.angular_velocity_mu_derivatives = []
        for component in self.components:
            self.velocity_mass_derivatives.append(np.array([0., 0., 0.]))
            self.angular_velocity_mass_derivatives.append(np.array([0., 0., 0.]))
            self.velocity_mu_derivatives.append(np.array([0., 0., 0.]))
            self.angular_velocity_mu_derivatives.append(np.array([0., 0., 0.]))

    def set_component_velocities_and_angular_velocities(self):
        for i in np.arange(len(self.components)):
            shape = self.components[i]
            displacement = self.component_translations[i]
            displacement_w = geometry_utils.to_world_coords(self, displacement)
            shape_velocity_w = geometry_utils.velocity_of_point(self, displacement_w)
            shape.velocity = shape_velocity_w
            shape.angular_velocity = self.angular_velocity

    def set_component_locations_and_orientations(self):
        for i in np.arange(len(self.components)):
            shape = self.components[i]
            displacement = self.component_translations[i]
            shape.location = geometry_utils.to_world_coords(self, displacement)
            shape.orientation = geometry_utils.quaternion_mult(self.component_rotations[i], self.orientation)

def set_up_component_shapes(info, location, orientation, masses, shape_ground_frictions_in):
    shapes = []
    center = np.array([0.,0.,0.])
    for coords in info:
        x,y,z = coords
        local_loc = np.array([x, y, z])
        center += local_loc
        shapes.append(collision_shape("box", (1.,1.,1.), local_loc, orientation, np.array([0., 0., 0.]), np.array([0., 0., 0.]), 1.))

    center /= len(info)
    for a_shape in shapes:
        a_shape.location -= center
    for a_shape in shapes:
        a_shape.location = geometry_utils.rotation_from_quaternion(orientation, a_shape.location) + location


    for count,shape in enumerate(shapes):
        shape.mass = masses[count]
        shape.I = masses[count]*shape.I_mass_derivative
        shape.I_inv = np.linalg.inv(shape.I)
        shape.I_inv_mass_derivatives = [-1 * shape.I_inv * shape.I_inv * shape.I_mass_derivative]  # multiply component by component

    # set coefficients of friction for all shapes
    shape_shape_frictions = {}
    '''count = 0
    for shape in shapes:
        this_shape_frictions = {}
        for shape2 in shapes[count+1:]:
            this_shape_frictions[shape2] = 0.5          #set friction coefficient to 0.5 for now
        shape_shape_frictions[shape] = this_shape_frictions
        count += 1'''
    shape_ground_frictions = {}
    for shape in shapes:
        shape_ground_frictions[shape] = 0.02  # set friction coefficient to 0.02 for now
    for count,shape in enumerate(shapes):
        shape_ground_frictions[shape] = shape_ground_frictions_in[count]

    return shapes, shape_shape_frictions, shape_ground_frictions




def run_one_time_step(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, fixed_contact_shapes, find_derivatives, locations_records=None, motion_script_to_write=None, energies_records=None, time=None, script_at_current_step=None):
    if script_at_current_step is not None:
        combined.location[0] = script_at_current_step[1]
        combined.location[1] = script_at_current_step[2]
        combined.location[2] = script_at_current_step[3]
        combined.orientation[0] = script_at_current_step[4]
        combined.orientation[1] = script_at_current_step[5]
        combined.orientation[2] = script_at_current_step[6]
        combined.orientation[3] = script_at_current_step[7]
        combined.velocity[0] = script_at_current_step[8]
        combined.velocity[1] = script_at_current_step[9]
        combined.velocity[2] = script_at_current_step[10]
        combined.angular_velocity[0] = script_at_current_step[11]
        combined.angular_velocity[1] = script_at_current_step[12]
        combined.angular_velocity[2] = script_at_current_step[13]

    combined.set_component_locations_and_orientations()

    combined.set_component_velocities_and_angular_velocities()

    # update energies
    total_KE = 0
    total_PE = 0
    total_energy = 0
    for shape in shapes:
        total_KE += shape.KE
        total_PE += shape.PE
        total_energy += shape.KE + shape.PE

    # record
    if locations_records is not None:
        file_handling.write_records_and_motion_script(locations_records, motion_script_to_write, energies_records, time, total_KE, total_PE, total_energy, combined, shapes)

    # move
    for shape in shapes:
        shape.location += dt * shape.velocity
        shape.orientation += dt * shape.orientation_derivative
        shape.orientation = geometry_utils.normalize(shape.orientation)
        shape.orientation_derivative = geometry_utils.orientation_derivative(shape.orientation, shape.angular_velocity)
        shape.KE = 0.5 * shape.mass * np.square(np.linalg.norm(shape.velocity)) + 0.5 * np.dot(np.matmul(shape.I, shape.angular_velocity), shape.angular_velocity)
        shape.PE = 0
    combined.location += dt * combined.velocity
    combined.orientation += dt * combined.orientation_derivative
    combined.orientation = geometry_utils.normalize(combined.orientation)
    combined.orientation_derivative = geometry_utils.orientation_derivative(combined.orientation, combined.angular_velocity)

    '''#apply external forces
    for shape in shapes:
        shape.velocity[1] -= 9.8*dt
    combined.velocity[1] -= 9.8*dt'''

    # read contact info from existing script, if such as script has been provided
    if script_at_current_step is not None:
        shape_shape_contacts_low_level = []
        ground_contacts_low_level = []
        i = 14
        while i < len(script_at_current_step):
            type_of_contact = script_at_current_step[i]
            if type_of_contact == "shape-shape_contact":
                shape1_index = script_at_current_step[i + 1]
                shape2_index = script_at_current_step[i + 2]
                shape1 = shapes[shape1_index]
                shape2 = shapes[shape2_index]
                contact_location = np.array([0., 0., 0.])
                contact_location[0] = script_at_current_step[i + 3]
                contact_location[1] = script_at_current_step[i + 4]
                contact_location[2] = script_at_current_step[i + 5]
                normal = np.array([0., 0., 0.])
                normal[0] = script_at_current_step[i + 6]
                normal[1] = script_at_current_step[i + 7]
                normal[2] = script_at_current_step[i + 8]
                shape_shape_contacts_low_level.append(((shape1, shape2), (contact_location, normal)))
                i = i + 9
            elif type_of_contact == "ground-shape_contact":
                shape_index = script_at_current_step[i + 1]
                shape = shapes[shape_index]
                contact_location = np.array([0., 0., 0.])
                contact_location[0] = script_at_current_step[i + 2]
                contact_location[1] = script_at_current_step[i + 3]
                contact_location[2] = script_at_current_step[i + 4]
                normal = np.array([0., 0., 0.])
                normal[0] = script_at_current_step[i + 5]
                normal[1] = script_at_current_step[i + 6]
                normal[2] = script_at_current_step[i + 7]
                ground_contacts_low_level.append((shape, (contact_location, normal)))
                i = i + 8
    else:
        # update bounding boxes
        # bounding boxes consist of min_x,max_x,min_y,max_y,min_z,max_z
        for shape in shapes:
            if shape.shape_type == "box":
                world_vertices = []
                for vertex in shape.vertices:
                    world_vertices.append(geometry_utils.to_world_coords(shape, vertex))
                shape.bounding_box = geometry_utils.get_extrema(world_vertices)
            elif shape.shape_type == "sphere":
                shape.bounding_box = shape.location[0] - shape.radius, shape.location[0] + shape.radius, shape.location[1] - shape.radius, shape.location[1] + shape.radius, shape.location[2] - shape.radius, shape.location[2] + shape.radius
            elif shape.shape_type == "uncapped cylinder":
                shape.bounding_box = geometry_utils.get_cylinder_bounding_box_extrema(shape)

        # preliminary collision detection using bounding boxes
        collision_check_list = []
        ground_contacts_check_list = []
        for i in np.arange(len(shapes)):
            for j in np.arange(i + 1, len(shapes)):
                if (shapes[i], shapes[j]) in fixed_contact_shapes:
                    continue
                if (collision_detection.AABB_intersect(shapes[i], shapes[j])):
                    collision_check_list.append((shapes[i], shapes[j]))
            # check for ground collisions
            if shapes[i].bounding_box[2] < -0.00001:  # threshold
                ground_contacts_check_list.append(shapes[i])

        # main collision detection
        shape_shape_contacts_low_level = []  # no shape-shape collisions   collision_detection.shape_shape_collision_detection(collision_check_list)
        ground_contacts_low_level = collision_detection.shape_ground_collision_detection(ground_contacts_check_list)  # []#no ground contacts

    # write down the contacts in the motion script
    if locations_records is not None:
        file_handling.write_contacts_in_motion_script(shapes, shape_shape_contacts_low_level, ground_contacts_low_level, motion_script_to_write)

    # get friction coefficients
    shape_shape_contact_friction_coefficients = []
    ground_contact_friction_coefficients = []
    for i in np.arange(len(shape_shape_contacts_low_level)):
        shape_pair, contact = shape_shape_contacts_low_level[i]
        shape1, shape2 = shape_pair
        shape_shape_contact_friction_coefficients.append(shape_shape_frictions[shape1][shape2])
    for i in np.arange(len(ground_contacts_low_level)):
        shape, contact = ground_contacts_low_level[i]
        ground_contact_friction_coefficients.append(shape_ground_frictions[shape])

    #handle collisions
    collision_handling_basic_impulse.handle_collisions_using_impulses(shapes, ground_contacts_low_level, find_derivatives, dt, ground_contact_friction_coefficients)


def run_sim(start_time, dt, total_time, shapes, combined, shape_shape_frictions, shape_ground_frictions, writing_to_files, find_derivatives, existing_motion_script=None):
    time = start_time
    fixed_contact_shapes = combined.fixed_contacts

    if writing_to_files:
        locations_records, energies_records, motion_script, loc_file, dir_name = file_handling.setup_records_and_motion_script(shapes)

    step=0
    while(time < total_time):
        if True:  # (step % 40 == 0):#
            print("simulation\tt =", time)

        script_at_current_step = None
        if existing_motion_script is not None:
            if step >= len(existing_motion_script):
                print("Error: Simulation running for longer than the provided motion script")
                exit()
            script_at_current_step = existing_motion_script[step]

        if writing_to_files:
            run_one_time_step(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, fixed_contact_shapes, find_derivatives, locations_records=locations_records, motion_script_to_write=motion_script, energies_records=energies_records, time=time, script_at_current_step=script_at_current_step)
        else:
            run_one_time_step(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, fixed_contact_shapes, find_derivatives, script_at_current_step=script_at_current_step)

        #update time
        #to do: make this symplectic (velocity verlet)
        time+=dt
        step+=1

    if writing_to_files:
        locations_records.close()
        energies_records.close()
        motion_script.close()

        print("writing simulation files")
        file_handling.write_simulation_files(shapes, shape_ground_frictions, loc_file, dir_name, dt, 24)



def run_time_step_to_take_deviation_and_derivatives(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, motion_script, time_step):
    fixed_contact_shapes = combined.fixed_contacts

    script_at_current_step = motion_script[time_step]
    next_time_step = time_step + 1
    run_one_time_step(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, fixed_contact_shapes, True, script_at_current_step=script_at_current_step)

    deviations_from_script = []

    '''deviations_from_script.append(combined.location[0] - motion_script[next_time_step][1])
    deviations_from_script.append(combined.location[1] - motion_script[next_time_step][2])
    deviations_from_script.append(combined.location[2] - motion_script[next_time_step][3])
    deviations_from_script.append(combined.orientation[0] - motion_script[next_time_step][4])
    deviations_from_script.append(combined.orientation[1] - motion_script[next_time_step][5])
    deviations_from_script.append(combined.orientation[2] - motion_script[next_time_step][6])
    deviations_from_script.append(combined.orientation[3] - motion_script[next_time_step][7])'''
    deviations_from_script.append(combined.velocity[0] - motion_script[next_time_step][8])
    deviations_from_script.append(combined.velocity[1] - motion_script[next_time_step][9])
    deviations_from_script.append(combined.velocity[2] - motion_script[next_time_step][10])
    deviations_from_script.append(combined.angular_velocity[0] - motion_script[next_time_step][11])
    deviations_from_script.append(combined.angular_velocity[1] - motion_script[next_time_step][12])
    deviations_from_script.append(combined.angular_velocity[2] - motion_script[next_time_step][13])

    deviations_from_script_squared = []
    for deviation_from_script in deviations_from_script:
        deviations_from_script_squared.append(deviation_from_script*deviation_from_script)

    deviations_from_script_squared_velocity_mass_derivatives = []
    deviations_from_script_squared_angular_velocity_mass_derivatives = []
    deviations_from_script_squared_velocity_mu_derivatives = []
    deviations_from_script_squared_angular_velocity_mu_derivatives = []
    for shape_count,component in enumerate(combined.components):
        this_shape_velocity_mass_derivative = np.array([0., 0., 0.])
        this_shape_velocity_mass_derivative[0] = 2 * deviations_from_script[0] * combined.velocity_mass_derivatives[shape_count][0]
        this_shape_velocity_mass_derivative[1] = 2 * deviations_from_script[1] * combined.velocity_mass_derivatives[shape_count][1]
        this_shape_velocity_mass_derivative[2] = 2 * deviations_from_script[2] * combined.velocity_mass_derivatives[shape_count][2]
        deviations_from_script_squared_velocity_mass_derivatives.append(this_shape_velocity_mass_derivative)

        this_shape_angular_velocity_mass_derivative = np.array([0., 0., 0.])
        this_shape_angular_velocity_mass_derivative[0] = 2 * deviations_from_script[3] * combined.angular_velocity_mass_derivatives[shape_count][0]
        this_shape_angular_velocity_mass_derivative[1] = 2 * deviations_from_script[4] * combined.angular_velocity_mass_derivatives[shape_count][1]
        this_shape_angular_velocity_mass_derivative[2] = 2 * deviations_from_script[5] * combined.angular_velocity_mass_derivatives[shape_count][2]
        deviations_from_script_squared_angular_velocity_mass_derivatives.append(this_shape_angular_velocity_mass_derivative)

        this_shape_velocity_mu_derivative = np.array([0., 0., 0.])
        this_shape_velocity_mu_derivative[0] = 2 * deviations_from_script[0] * combined.velocity_mu_derivatives[shape_count][0]
        this_shape_velocity_mu_derivative[1] = 2 * deviations_from_script[1] * combined.velocity_mu_derivatives[shape_count][1]
        this_shape_velocity_mu_derivative[2] = 2 * deviations_from_script[2] * combined.velocity_mu_derivatives[shape_count][2]
        deviations_from_script_squared_velocity_mu_derivatives.append(this_shape_velocity_mu_derivative)

        this_shape_angular_velocity_mu_derivative = np.array([0., 0., 0.])
        this_shape_angular_velocity_mu_derivative[0] = 2 * deviations_from_script[3] * combined.angular_velocity_mu_derivatives[shape_count][0]
        this_shape_angular_velocity_mu_derivative[1] = 2 * deviations_from_script[4] * combined.angular_velocity_mu_derivatives[shape_count][1]
        this_shape_angular_velocity_mu_derivative[2] = 2 * deviations_from_script[5] * combined.angular_velocity_mu_derivatives[shape_count][2]
        deviations_from_script_squared_angular_velocity_mu_derivatives.append(this_shape_angular_velocity_mu_derivative)

    #sum up the deviations and derivatives
    sum_deviations_from_script_squared = 0.
    for deviation_from_script_squared in deviations_from_script_squared:
        sum_deviations_from_script_squared += deviation_from_script_squared
    sum_deviations_from_script_squared_mass_derivatives = []
    sum_deviations_from_script_squared_mu_derivatives = []
    for shape_count, component in enumerate(combined.components):
        sum_deviations_from_script_squared_mass_derivative = 0.
        sum_deviations_from_script_squared_mu_derivative = 0.
        for i in np.arange(3):
            sum_deviations_from_script_squared_mass_derivative += deviations_from_script_squared_velocity_mass_derivatives[shape_count][i]
            sum_deviations_from_script_squared_mass_derivative += deviations_from_script_squared_angular_velocity_mass_derivatives[shape_count][i]
            sum_deviations_from_script_squared_mu_derivative += deviations_from_script_squared_velocity_mu_derivatives[shape_count][i]
            sum_deviations_from_script_squared_mu_derivative += deviations_from_script_squared_angular_velocity_mu_derivatives[shape_count][i]
        sum_deviations_from_script_squared_mass_derivatives.append(sum_deviations_from_script_squared_mass_derivative)
        sum_deviations_from_script_squared_mu_derivatives.append(sum_deviations_from_script_squared_mu_derivative)

    #reset the shape's derivatives
    combined.reset_velocity_and_angular_velocity_derivatives()

    return sum_deviations_from_script_squared, sum_deviations_from_script_squared_mass_derivatives, sum_deviations_from_script_squared_mu_derivatives


def run_mass_derivatives_sweep_in_combined_shape(shape_to_alter_index, values_to_count, shape_masses, shape_ground_frictions_in):
    result = []
    result_deriv = []
    result_deriv_estimate = []

    outermost_count = 0

    for value in values_to_count:
        # make shapes
        shape_masses[shape_to_alter_index] = value
        shapes, shape_shape_frictions, shape_ground_frictions = \
            set_up_component_shapes(combined_info, np.array([0., 0., 0.]), np.array([0., 0., 0., 1.]), shape_masses, shape_ground_frictions_in)
        combined = combined_body(shapes, np.array([0., 0., 0.]), np.array([0., 0., 0.]))

        #append the results to the list
        result.append(combined.I_inv[0][0])
        result_deriv.append(combined.I_inv_mass_derivatives[shape_to_alter_index][0][0])

        if outermost_count != 0:
            result_deriv_estimate.append((result[outermost_count]-result[outermost_count - 1]) / (value - values_to_count[outermost_count - 1]))
        else:
            result_deriv_estimate.append(0)
        outermost_count += 1

    #draw plot
    draw_data.plot_data(values_to_count, result, result_deriv, result_deriv_estimate)

    #print out errors
    avg_abs_error = 0.
    for i in range(1, len(values_to_count)):
        abs_error = np.abs(result_deriv_estimate[i] - result_deriv[i])
        avg_abs_error += abs_error
    avg_abs_error /= len(values_to_count) - 1
    print()
    print("derivatives avg error =",avg_abs_error)


def run_derivatives_sweep(shape_to_alter_index, doing_friction, values_to_count, motion_script, time_step, shape_masses, shape_ground_frictions_in):
    result = []
    result_deriv = []
    result_deriv_estimate = []

    outermost_count = 0


    for value in values_to_count:
        # make shapes
        if doing_friction:
            shape_ground_frictions_in[shape_to_alter_index] = value
        else:
            shape_masses[shape_to_alter_index]=value
        shapes, shape_shape_frictions, shape_ground_frictions = \
            set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation, shape_masses, shape_ground_frictions_in)
        combined = combined_body(shapes, np.array([-1., 0., 0.]), np.array([0., 0., 0.]))

        deviation_from_script_squared, \
        deviations_from_script_squared_mass_derivatives, \
        deviations_from_script_squared_mu_derivatives = \
            run_time_step_to_take_deviation_and_derivatives(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, motion_script, time_step)

        #append the results to the list
        result.append(deviation_from_script_squared)
        if doing_friction:
            result_deriv.append(deviations_from_script_squared_mu_derivatives[shape_to_alter_index])
        else:
            result_deriv.append(deviations_from_script_squared_mass_derivatives[shape_to_alter_index])

        if outermost_count != 0:
            result_deriv_estimate.append((result[outermost_count]-result[outermost_count - 1]) / (value - values_to_count[outermost_count - 1]))
        else:
            result_deriv_estimate.append(0)
        outermost_count += 1

    #draw plot
    draw_data.plot_data(values_to_count, result, result_deriv, result_deriv_estimate)

    #print out errors
    avg_abs_error = 0.
    '''avg_rel_error = 0.
    rel_error_available_count = 0'''
    for i in range(1, len(values_to_count)):
        abs_error = np.abs(result_deriv_estimate[i] - result_deriv[i])
        avg_abs_error += abs_error
        '''rel_error = 0.
        if abs(result_deriv_estimate[i]) > 0.00001: #threshold
            rel_error = abs_error / result_deriv_estimate[i]
            avg_rel_error += rel_error
            rel_error_available_count += 1'''
    avg_abs_error /= len(values_to_count) - 1
    print()
    print("derivatives avg error =",avg_abs_error)
    '''if rel_error_available_count > 0:
        avg_rel_error /= rel_error_available_count
        print("derivatives avg relative error =", avg_rel_error)
    else:
        print("derivatives avg relative error is not available, since estimated derivatives are zero or close to zero")'''

def run_2D_derivatives_sweep(shape_to_alter_index1, shape_to_alter_index2, doing_friction, values_to_count, motion_script, time_step, shape_masses, shape_ground_frictions_in):
    X, Y = np.meshgrid(values_to_count, values_to_count)

    zero = X*0
    Z_result = X*0
    Z_x_derivative = X*0
    Z_y_derivative = X*0
    Z_x_derivative_estimate = X*0
    Z_y_derivative_estimate = X*0
    for coord_x, value_x in enumerate(values_to_count):
        for coord_y, value_y in enumerate(values_to_count):
            # make shapes
            if doing_friction:
                shape_ground_frictions_in[shape_to_alter_index1] = value_x
                shape_ground_frictions_in[shape_to_alter_index2] = value_y
            else:
                shape_masses[shape_to_alter_index1] = value_x
                shape_masses[shape_to_alter_index2] = value_y
            shapes, shape_shape_frictions, shape_ground_frictions = \
                set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation, shape_masses, shape_ground_frictions_in)
            combined = combined_body(shapes, np.array([-1., 0., 0.]), np.array([0., 0., 0.]))

            Z_result[coord_x, coord_y], mass_derivatives, mu_derivatives = \
                run_time_step_to_take_deviation_and_derivatives(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, motion_script, time_step)
            if doing_friction:
                Z_x_derivative[coord_x, coord_y] = mu_derivatives[shape_to_alter_index1]
                Z_y_derivative[coord_x, coord_y] = mu_derivatives[shape_to_alter_index2]
            else:
                Z_x_derivative[coord_x, coord_y] = mass_derivatives[shape_to_alter_index1]
                Z_y_derivative[coord_x, coord_y] = mass_derivatives[shape_to_alter_index2]
            Z_x_derivative_estimate[coord_x, coord_y] = (Z_result[coord_x, coord_y] - Z_result[coord_x - 1, coord_y]) / (values_to_count[coord_x] - values_to_count[coord_x - 1])
            Z_y_derivative_estimate[coord_x, coord_y] = (Z_result[coord_x, coord_y] - Z_result[coord_x, coord_y - 1]) / (values_to_count[coord_y] - values_to_count[coord_y - 1])

    # draw plot
    draw_data.plot_3D_data(X, Y, Z_result, Z_x_derivative, Z_x_derivative_estimate, zero)
    draw_data.plot_3D_data(X, Y, Z_result, Z_y_derivative, Z_y_derivative_estimate, zero)

def find_values(motion_script, time_step, initial_guess, bounds, shapes_len):
    #mu only. Add masses later.

    def func(x):
        # make shapes
        shapes, shape_shape_frictions, shape_ground_frictions = set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation, x[:shapes_len], x[shapes_len:])
        combined = combined_body(shapes, np.array([-1., 0., 0.]), np.array([0., 0., 0.]))

        result, mass_derivatives, mu_derivatives = \
            run_time_step_to_take_deviation_and_derivatives(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, motion_script, time_step)
        mult_factor = 10000
        return mult_factor*result, mult_factor*np.array([mass_derivatives, mu_derivatives]).flatten()
    result = scipy.optimize.minimize(func, x0 = initial_guess, method='L-BFGS-B', jac=True, bounds=bounds)

    return result.x

def ordinary_run(shape_masses, shape_ground_frictions_in, motion_script = None):
    # make shapes
    shapes, shape_shape_frictions, shape_ground_frictions =  set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation, shape_masses, shape_ground_frictions_in)
    combined = combined_body(shapes, np.array([-1., 0., 0.]), np.array([0., 0., 0.]))

    # set time
    time = 0
    total_time = 10

    # run the simulation
    run_sim(time, dt, total_time, shapes, combined, shape_shape_frictions,shape_ground_frictions, True, False, motion_script)



#load shape info
print()
#combined_info = file_handling.read_combined_boxes_rigid_body_file("try1.txt")
combined_info = file_handling.read_combined_boxes_rigid_body_file("hammer.txt")
rotation = geometry_utils.normalize(np.array([0., 0.3, 0., 0.95]))

# set dt
dt = 0.001


masses = np.linspace(0.5, 5., 1000)
mu_values = np.linspace(0, 0.5, 1000)

actual_mass_values = 1.*np.ones((len(combined_info)))
actual_mu_values = 0.02*np.ones((len(combined_info)))

##for 2-component shape:
#actual_mass_values[0]=3.
#for hammer
for i in np.arange(32):
    actual_mass_values[i] = 10.

time_step = 100
#motion_script = file_handling.read_motion_script_file(os.path.join("test5","motion_script.csv"))

ordinary_run(actual_mass_values, actual_mu_values)
#ordinary_run(actual_mass_values, actual_mu_values, motion_script=motion_script)
#run_derivatives_sweep(0, False, masses, motion_script, time_step, actual_mass_values, actual_mu_values)
#run_derivatives_sweep(0, True, mu_values, motion_script, time_step, actual_mass_values, actual_mu_values)
#run_2D_derivatives_sweep(0, 1, True, mu_values, motion_script, time_step, actual_mass_values, actual_mu_values)
'''shapes_len = len(combined_info)
#initial_guess = np.array([1., 1., 0.005, 0.4])
initial_guess = np.concatenate(np.random.random(len(combined_info)) - 0.5 + actual_mass_values, (np.random.random(len(combined_info)) - 0.5)/3 + actual_mu_values)
ordinary_run(initial_guess[:shapes_len], initial_guess[shapes_len:])
bounds = [(0.5, 5.), (0.5, 5.), (0., 0.5), (0., 0.5)]
vals = find_values(motion_script, time_step, initial_guess, bounds, shapes_len)
ordinary_run(vals[:shapes_len], vals[shapes_len:])
print(vals)'''

#run_mass_derivatives_sweep_in_combined_shape(0, masses, [1.,1.], [0.2, 0.02])
