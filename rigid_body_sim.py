import numpy as np
import scipy
import os
import collision_handling_basic_impulse
import collision_detection
#import collision_handling_LCP
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
        world_COM_times_mass = np.array([0., 0., 0.])
        self.location = np.array([0., 0., 0.])
        for shape in self.components:
            self.location += shape.location
            self.mass += shape.mass
            world_COM_times_mass += shape.mass*geometry_utils.to_world_coords(shape, shape.COM)
        self.location /= len(self.components)   #location is non-weighted average of component locations
        self.COM = world_COM_times_mass / self.mass - self.location

        #component rotations from this shape to child shape
        #component translations from this shape to child shape
        self.component_rotations = []
        self.component_translations = []
        self.component_translations_COM = []
        for shape in self.components:
            self.component_rotations.append(shape.orientation + 0.)
            self.component_translations.append(shape.location - self.location)
            self.component_translations_COM.append(shape.location - self.COM)

        #get moment of inertia and its inverse
        self.I = np.zeros((3,3))
        for index in np.arange(len(self.components)):
            shape = self.components[index]
            displacement = self.component_translations_COM[index]
            translated_shape_I = shape.I + shape.mass*(np.dot(displacement, displacement)*np.identity(3) - np.outer(displacement, displacement))
            R = geometry_utils.quaternion_to_rotation_matrix(self.component_rotations[index])
            self.I += np.matmul(R, np.matmul(translated_shape_I, R.T))
        self.I_inv = np.linalg.inv(self.I)

        #get mass derivatives of combined body's center of mass
        self.COM_mass_derivatives = []
        second_term = world_COM_times_mass / (self.mass * self.mass)
        for index in np.arange(len(self.components)):
            shape = self.components[index]
            self.COM_mass_derivatives.append(geometry_utils.to_world_coords(shape, shape.COM) / self.mass - second_term)
            #d_COM/d_mass = (d_world_COM_times_mass/d_mass) / self.mass - world_COM_times_mass / (self.mass * self.mass)
        #get mass derivatives of moment of inertia and its inverse, and of its location
        self.I_mass_derivatives = []
        for index in np.arange(len(self.components)):
            self.I_mass_derivatives.append(np.zeros((3,3)))
            shape = self.components[index]
            displacement = self.component_translations_COM[index]
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




def run_one_time_step(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, fixed_contact_shapes, find_derivatives, external_force_script_step, locations_records=None, motion_script_to_write=None, energies_records=None, time=None, motion_script_at_current_step=None):
    writing_motion_script = (locations_records is not None)

    if motion_script_at_current_step is not None:
        combined.location[0] = motion_script_at_current_step[1]
        combined.location[1] = motion_script_at_current_step[2]
        combined.location[2] = motion_script_at_current_step[3]
        combined.orientation[0] = motion_script_at_current_step[4]
        combined.orientation[1] = motion_script_at_current_step[5]
        combined.orientation[2] = motion_script_at_current_step[6]
        combined.orientation[3] = motion_script_at_current_step[7]
        combined.velocity[0] = motion_script_at_current_step[8]
        combined.velocity[1] = motion_script_at_current_step[9]
        combined.velocity[2] = motion_script_at_current_step[10]
        combined.angular_velocity[0] = motion_script_at_current_step[11]
        combined.angular_velocity[1] = motion_script_at_current_step[12]
        combined.angular_velocity[2] = motion_script_at_current_step[13]

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
    if writing_motion_script:
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

    '''#apply gravity explicitly
    for shape in shapes:
        shape.velocity[1] -= 9.8*dt
    combined.velocity[1] -= 9.8*dt'''

    # read contact info from existing script, if such as script has been provided
    if motion_script_at_current_step is not None:
        shape_shape_contacts_low_level = []
        ground_contacts_low_level = []
        i = 14
        while i < len(motion_script_at_current_step):
            type_of_contact = motion_script_at_current_step[i]
            if type_of_contact == "shape-shape_contact":
                shape1_index = motion_script_at_current_step[i + 1]
                shape2_index = motion_script_at_current_step[i + 2]
                shape1 = shapes[shape1_index]
                shape2 = shapes[shape2_index]
                contact_location = np.array([0., 0., 0.])
                contact_location[0] = motion_script_at_current_step[i + 3]
                contact_location[1] = motion_script_at_current_step[i + 4]
                contact_location[2] = motion_script_at_current_step[i + 5]
                normal = np.array([0., 0., 0.])
                normal[0] = motion_script_at_current_step[i + 6]
                normal[1] = motion_script_at_current_step[i + 7]
                normal[2] = motion_script_at_current_step[i + 8]
                shape_shape_contacts_low_level.append(((shape1, shape2), (contact_location, normal)))
                i = i + 9
            elif type_of_contact == "ground-shape_contact":
                shape_index = motion_script_at_current_step[i + 1]
                shape = shapes[shape_index]
                contact_location = np.array([0., 0., 0.])
                contact_location[0] = motion_script_at_current_step[i + 2]
                contact_location[1] = motion_script_at_current_step[i + 3]
                contact_location[2] = motion_script_at_current_step[i + 4]
                normal = np.array([0., 0., 0.])
                normal[0] = motion_script_at_current_step[i + 5]
                normal[1] = motion_script_at_current_step[i + 6]
                normal[2] = motion_script_at_current_step[i + 7]
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
            if shapes[i].bounding_box[2] < -0.00001:  # threshold, bounding_box[2] is min_y
                ground_contacts_check_list.append(shapes[i])

        # main collision detection
        shape_shape_contacts_low_level = []  # no shape-shape collisions   collision_detection.shape_shape_collision_detection(collision_check_list)
        ground_contacts_low_level = collision_detection.shape_ground_collision_detection(ground_contacts_check_list)  # []#no ground contacts

    # write down the contacts in the motion script
    if writing_motion_script:
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

    #apply external forces
    init_vel_x = combined.velocity[0]
    init_vel_y = combined.velocity[1]
    init_vel_z = combined.velocity[2]
    init_vel = np.linalg.norm(combined.velocity)
    #print(init_vel)
    external_force_magn, component_number, direction_x = external_force_script_step
    external_force_contact_location = collision_handling_basic_impulse.external_force_impulse(combined, external_force_magn, component_number, direction_x, dt, find_derivatives)
    #combined.set_component_velocities_and_angular_velocities()

    if writing_motion_script:
        file_handling.write_external_force_info_in_motion_script(external_force_magn, external_force_contact_location, direction_x, motion_script_to_write)

    #handle collisions
    #collision_handling_LCP.handle_collisions_LCP(combined, ground_contacts_low_level, dt)
    push_vel_x= combined.velocity[0]
    push_vel_y= combined.velocity[1]
    push_vel_z= combined.velocity[2]
    push_vel = np.linalg.norm(combined.velocity)
    #print("push_vel",push_vel)
    '''print("init_momentum",combined.mass*np.array([init_vel_x,init_vel_y,init_vel_z]))
    print("push_momentum",combined.mass*np.array([push_vel_x,push_vel_y,push_vel_z]))
    net_momentum =np.array([0.,0.,0.])
    for i in np.arange(len(ground_contacts_low_level)):
        shape, contact = ground_contacts_low_level[i]
        world_point, normal = contact
        tangential_velocity = collision_handling_basic_impulse.get_shape_ground_tangential_velocity(shape, world_point, normal)
        net_momentum += shape.mass*tangential_velocity
    print("net_momentum",net_momentum)
    net_momentum =np.array([0.,0.,0.])
    for shape in combined.components:
        net_momentum += shape.mass * shape.velocity
    print("net_momentum",net_momentum)'''
    #collision_handling_basic_impulse.handle_collisions_using_impulses(shapes, ground_contacts_low_level, find_derivatives, dt, ground_contact_friction_coefficients)

    final_vel_x = combined.velocity[0]
    final_vel_y = combined.velocity[1]
    final_vel_z = combined.velocity[2]
    final_vel = np.linalg.norm(combined.velocity)
    #print(final_vel)

    print()
    print(external_force_magn)
    #print(np.array([push_vel_x, push_vel_y, push_vel_z]) - np.array([init_vel_x, init_vel_y, init_vel_z]))
    #print(np.dot(np.array([init_vel_x, init_vel_y, init_vel_z]), np.array([direction_x, 0., 0.])))
    print(np.array([push_vel_x, push_vel_y, push_vel_z]))
    print(np.array([final_vel_x, final_vel_y, final_vel_z]))
    print(np.array([final_vel_x-push_vel_x, final_vel_y-push_vel_y, final_vel_z-push_vel_z]))
    #print(1./((push_vel_x-init_vel_x)/dt/external_force_magn))
    #print(final_vel_x-push_vel_x)
    print("\n\n")


def run_sim(start_time, dt, total_time, shapes, combined, shape_shape_frictions, shape_ground_frictions, writing_to_files, find_derivatives, existing_motion_script=None):
    time = start_time
    fixed_contact_shapes = combined.fixed_contacts

    if writing_to_files:
        locations_records, energies_records, motion_script, loc_file, dir_name = file_handling.setup_records_and_motion_script(shapes)

    step=0
    while(time < total_time):
        if True:  # (step % 40 == 0):#
            print("simulation\tt =", time)

        motion_script_at_current_step = None
        if existing_motion_script is not None:
            if step >= len(existing_motion_script):
                print("Error: Simulation running for longer than the provided motion script")
                exit()
            motion_script_at_current_step = existing_motion_script[step]

        if step >= len(external_force_script):
            external_force_script_step = [0., 0, 0]
        else:
            external_force_script_step = external_force_script[step]

        if writing_to_files:
            run_one_time_step(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, fixed_contact_shapes, find_derivatives, external_force_script_step, locations_records=locations_records, motion_script_to_write=motion_script, energies_records=energies_records, time=time, motion_script_at_current_step=motion_script_at_current_step)
        else:
            run_one_time_step(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, fixed_contact_shapes, find_derivatives, external_force_script_step, motion_script_at_current_step=motion_script_at_current_step)

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

    external_force_script_step = external_force_script[time_step]

    motion_script_at_current_step = motion_script[time_step]
    next_time_step = time_step + 1
    run_one_time_step(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, fixed_contact_shapes, True, external_force_script_step, motion_script_at_current_step=motion_script_at_current_step)

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

    #print("\n\ndeviations_from_script", deviations_from_script)
    #print("velocity mass derivatives", combined.velocity_mass_derivatives)
    #print("velocity mu derivatives", combined.velocity_mu_derivatives)

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
    draw_data.plot_data_three_curves(values_to_count, result, result_deriv, result_deriv_estimate)

    #print out errors
    avg_abs_error = 0.
    for i in range(1, len(values_to_count)):
        abs_error = np.abs(result_deriv_estimate[i] - result_deriv[i])
        avg_abs_error += abs_error
    avg_abs_error /= len(values_to_count) - 1
    print()
    print("derivatives avg error =",avg_abs_error)


def run_derivatives_sweep(shape_to_alter_index, doing_friction, values_to_count, motion_script, external_force_script, time_step, shape_masses, shape_ground_frictions_in):
    result = []
    result_deriv = []
    result_deriv_estimate = []

    outermost_count = 0


    for value in values_to_count:
        print("now on",value)
        # make shapes
        if doing_friction:
            shape_ground_frictions_in[shape_to_alter_index] = value
        else:
            shape_masses[shape_to_alter_index]=value
        shapes, shape_shape_frictions, shape_ground_frictions = \
            set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation, shape_masses, shape_ground_frictions_in)
        combined = combined_body(shapes, np.array([-0., 0., 0.]), np.array([0., 0., 0.]))

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

def run_2D_derivatives_sweep(shape_to_alter_index1, shape_to_alter_index2, doing_friction, values_to_count, motion_script, external_force_script, time_step, shape_masses, shape_ground_frictions_in):
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
            combined = combined_body(shapes, np.array([-0., 0., 0.]), np.array([0., 0., 0.]))

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

def run_2D_loss_sweep(shape_to_alter_index1, shape_to_alter_index2, doing_friction, values_to_count_index1, values_to_count_index2, motion_script, external_force_script, time_step, shape_masses, shape_ground_frictions_in):
    X, Y = np.meshgrid(values_to_count_index1, values_to_count_index2)
    X=X.T
    Y=Y.T

    zero = X*0
    Z_result = X*0
    print("start")
    for coord_x, value_x in enumerate(values_to_count_index1):
        for coord_y, value_y in enumerate(values_to_count_index2):
            # make shapes
            if doing_friction:
                shape_ground_frictions_in[shape_to_alter_index1] = value_x
                shape_ground_frictions_in[shape_to_alter_index2] = value_y
            else:
                shape_masses[shape_to_alter_index1] = value_x
                shape_masses[shape_to_alter_index2] = value_y
            shapes, shape_shape_frictions, shape_ground_frictions = \
                set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation, shape_masses, shape_ground_frictions_in)
            #shapes, shape_shape_frictions, shape_ground_frictions = \
            #    set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation,
            #                            np.concatenate((np.repeat(shape_masses[shape_to_alter_index1], 32), np.repeat(shape_masses[shape_to_alter_index2], 56))),
            #                            np.concatenate((np.repeat(shape_ground_frictions_in[shape_to_alter_index1], 32), np.repeat(shape_ground_frictions_in[shape_to_alter_index2], 56))))

            combined = combined_body(shapes, np.array([-0., 0., 0.]), np.array([0., 0., 0.]))
            print(coord_x, coord_y)
            print(Z_result.shape)
            Z_result[coord_x, coord_y], mass_derivatives, mu_derivatives = \
                run_time_step_to_take_deviation_and_derivatives(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, motion_script, time_step)

            #to help with log
            if Z_result[coord_x, coord_y]==0.:
                Z_result[coord_x, coord_y] = 1.e-16

            #loss_sweep_file.write(str(value_x)+","+str(value_y)+","+str(np.log10(Z_result[coord_x, coord_y]))+"\n")

        print("coord_x:",coord_x)

    # draw plot
    draw_data.plot_3D_loss_func(X.T, Y.T, np.log10(Z_result.T))

def find_values_L_BFSG_B(motion_script, external_force_script, time_step, initial_guess, bounds, shapes_len):

    def func(x):
        # make shapes
        #shapes, shape_shape_frictions, shape_ground_frictions = set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation, 19.5*x[:shapes_len]+0.5, x[shapes_len:]/2.)

        #hammer with only head and handle
        shapes, shape_shape_frictions, shape_ground_frictions = set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation,
                                                                                        np.concatenate((np.repeat(376./32.*(.8*x[0]+.1),32), np.repeat(376./56.*(1.-(.8*x[0]+.1)),56))),
                                                                                        np.concatenate((np.repeat(x[2]/2.,32), np.repeat(x[3]/2.,56))))
        combined = combined_body(shapes, np.array([-0., 0., 0.]), np.array([0., 0., 0.]))

        result, mass_derivatives, mu_derivatives = \
            run_time_step_to_take_deviation_and_derivatives(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, motion_script, time_step)
        '''#do for another 24 time steps
        for i in range(1,25):
            d_result, d_mass_derivatives, d_mu_derivatives = \
                run_time_step_to_take_deviation_and_derivatives(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, motion_script, time_step+i)
            result += d_result
            for j in range(shapes_len):
                mass_derivatives[j] += d_mass_derivatives[j]
                mu_derivatives[j] += d_mu_derivatives[j]
        if friction_only:
            for i in range(shapes_len):
                mass_derivatives[i] = 0.
        if mass_only:
            for i in range(shapes_len):
                mu_derivatives[i] = 0.'''

        #hammer with only head and handle
        mass_derivatives_head = np.average(mass_derivatives[:32])
        mass_derivatives_handle = np.average(mass_derivatives[32:])
        mu_derivatives_head = np.average(mu_derivatives[:32])
        mu_derivatives_handle = np.average(mu_derivatives[32:])
        derivatives = np.array([376./32.*.8*mass_derivatives_head - 376./56.*.8*mass_derivatives_handle, 0., mu_derivatives_head/2., mu_derivatives_handle/2.]).flatten()

        #derivatives = np.concatenate((19.5*np.array([mass_derivatives]), np.array([mu_derivatives])/2.)).flatten()

        print("before ln: result:",result,"\t\tderivatives:",derivatives)
        derivatives = derivatives / result
        result = np.log(result)

        print("ln(loss):",result)
        print("normalized masses:",x[0],1.-x[0],"\t\tnormalized mass derivatives:",derivatives[0],derivatives[1])
        print("normalized mu vals:",x[2],x[3],"\t\tnormalized mu derivatives:",derivatives[2],derivatives[3])
        print("masses:",((.8*x[0]+.1) * 376. / 32.),((1. - (.8*x[0]+.1)) * 376. / 56.))
        print("mu vals:",x[2] / 2., x[3] / 2.)
        print("normalized mass*friction:", x[0] * x[2], "\t\t", (1. - x[0]) * x[3])
        print("mass*friction:", ((.8*x[0]+.1) * 376. / 32.) * x[2] / 2., "\t\t", ((1. - (.8*x[0]+.1)) * 376. / 56.) * x[3] / 2.)
        print("mass sum",combined.mass)
        print("mass*friction/(mass sum)",((.8*x[0]+.1)*376./32.)*x[2]/2./combined.mass, ((1.-(.8*x[0]+.1))*376./56.)*x[3]/2/combined.mass)
        print()
        #return mult_factor*result, mult_factor*derivatives
        return result, derivatives

    global mult_factor
    friction_only=False
    mass_only=False
    result = scipy.optimize.minimize(func, x0 = initial_guess, method='L-BFGS-B', jac=True, bounds=bounds)#, options={'ftol':-30.})

    print("result.status", result.status, "result.success", result.success, "number of iterations", result.nit)
    print("log(loss) function final value:", result.fun)
    print("log(loss) function derivatives:\n",result.jac)
    print("final result:\n")
    print(str(result.fun) + "," + str(result.x[0]) + "," + str(1.-result.x[0]) + "," + str(result.x[2]) + "," + str(result.x[3]) + ",," + str(result.jac[0]) + "," + str(result.jac[1]) + "," + str(result.jac[2]) + "," + str(result.jac[3]) + "\n")
    return result.x

'''def find_values(motion_script, external_force_script, time_step, initial_guess, bounds, shapes_len):

    def func(x):
        # make shapes
        #shapes, shape_shape_frictions, shape_ground_frictions = set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation, x[:shapes_len], x[shapes_len:])

        #hammer with only head and handle
        shapes, shape_shape_frictions, shape_ground_frictions = set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation,
                                                                                        np.concatenate((np.repeat(376./32.*x[0],32), np.repeat(376./56.*(1.-x[0]),56))),
                                                                                        np.concatenate((np.repeat(x[2]/2.,32), np.repeat(x[3]/2.,56))))
        combined = combined_body(shapes, np.array([-0., 0., 0.]), np.array([0., 0., 0.]))

        result, mass_derivatives, mu_derivatives = \
            run_time_step_to_take_deviation_and_derivatives(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, motion_script, time_step)
        ##do for another 24 time steps
        #for i in range(1,25):
        #    d_result, d_mass_derivatives, d_mu_derivatives = \
        #        run_time_step_to_take_deviation_and_derivatives(dt, shapes, combined, shape_shape_frictions, shape_ground_frictions, motion_script, time_step+i)
        #    result += d_result
        #    for j in range(shapes_len):
        #        mass_derivatives[j] += d_mass_derivatives[j]
        #        mu_derivatives[j] += d_mu_derivatives[j]
        #if friction_only:
        #    for i in range(shapes_len):
        #        mass_derivatives[i] = 0.

        #hammer with only head and handle
        mass_derivatives_head = np.average(mass_derivatives[:32])
        #mass_derivatives_handle = np.average(mass_derivatives[32:])
        mu_derivatives_head = np.average(mu_derivatives[:32])
        mu_derivatives_handle = np.average(mu_derivatives[32:])
        derivatives = np.array([376./32.*mass_derivatives_head, 0., mu_derivatives_head/2., mu_derivatives_handle/2.]).flatten()

        #derivatives = np.array([mass_derivatives, mu_derivatives]).flatten()

        print("before ln: result:",result,"\t\tderivatives:",derivatives)
        derivatives = derivatives / result
        result = np.log(result)

        print("ln(loss):", result)
        print("normalized masses:",x[0],1.-x[0],"\t\tnormalized mass derivatives:",derivatives[0],derivatives[1])
        print("normalized mu vals:",x[2],x[3],"\t\tnormalized mu derivatives:",derivatives[2],derivatives[3])
        print("normalized mass*friction:",x[0]*x[2],"\t\t",(1.-x[0])*x[3])
        print("mass*friction:",(x[0]*376./32.)*x[2]/2.,"\t\t",((1.-x[0])*376./56.)*x[3]/2.)
        print("mass sum",combined.mass)
        print("mass*friction/(mass sum)",(x[0]*376./32.)*x[2]/2./combined.mass, ((1.-x[0])*376./56.)*x[3]/2/combined.mass)
        print()
        return result, derivatives

    friction_only=False
    iteration = 0
    loss = 0.
    tolerance = -30
    start=True
    #mass_derivatives_mult_factor = 1.
    #mu_derivatives_mult_factor = 1.
    general_mult_factor = 0.3
    x = initial_guess
    while loss>tolerance or start:
        print("iteration:",iteration)
        loss, derivatives = func(x)
        #mass_derivatives = derivatives[:shapes_len]
        #mu_derivatives = derivatives[shapes_len:]
        mass_derivatives = derivatives[:2]          #due to hammer with head and handle
        mu_derivatives = derivatives[2:]          #due to hammer with head and handle
        if start:
            start = False
            #mass_derivatives_mult_factor = 1. / np.linalg.norm(mass_derivatives)
            #mu_derivatives_mult_factor = 1. / np.linalg.norm(mu_derivatives)
            #print("mass_derivatives_mult_factor",mass_derivatives_mult_factor)
            #print("mu_derivatives_mult_factor",mu_derivatives_mult_factor)
        #for i in np.arange(shapes_len):
        for i in np.arange(2):          #due to hammer with head and handle
            step_mass = (bounds[i][1]-bounds[i][0]) * general_mult_factor * np.power(0.999,iteration) * 10*np.sign(mass_derivatives[i])#mass_derivatives_mult_factor*mass_derivatives[i]
            #step_mu = (bounds[i+shapes_len][1]-bounds[i+shapes_len][0]) * 0.1 * np.power(0.999,iteration) * np.sign(mu_derivatives[i]#mu_derivatives_mult_factor*mu_derivatives[i]
            step_mu = (bounds[i+2][1]-bounds[i+2][0]) * general_mult_factor * np.power(0.999,iteration) * np.sign(mu_derivatives[i])#mu_derivatives_mult_factor*mu_derivatives[i] #due to hammer with head and handle

            x[i] -= step_mass
            # x[i+shapes_len] -= step_mu
            x[i + 2] -= step_mu #due to hammer with head and handle
            #print("\tscaled mass derivative",mass_derivatives_mult_factor*mass_derivatives[i])
            print("\tstep mass",-step_mass)
            #print("\tscaled mu derivative",mu_derivatives_mult_factor*mu_derivatives[i])
            print("\tstep mu",-step_mu)
            print()

            #clamp within bounds
            if x[i] > bounds[i][1]:
                x[i] = bounds[i][1]
            elif x[i] < bounds[i][0]:
                x[i] = bounds[i][0]
            #if x[i+shapes_len] > bounds[i+shapes_len][1]:
            #    x[i+shapes_len] = bounds[i+shapes_len][1]
            #elif x[i+shapes_len] < bounds[i+shapes_len][0]:
            #    x[i+shapes_len] = bounds[i+shapes_len][0]
            if x[i+2] > bounds[i+2][1]:                         #due to hammer with head and handle
                x[i+2] = bounds[i+2][1]
            elif x[i+2] < bounds[i+2][0]:
                x[i+2] = bounds[i+2][0]
        iteration += 1
        print()
    return x'''


def ordinary_run(shape_masses, shape_ground_frictions_in, motion_script = None):
    # make shapes
    shapes, shape_shape_frictions, shape_ground_frictions =  set_up_component_shapes(combined_info, np.array([0., 0.4999804, 5.]), rotation, shape_masses, shape_ground_frictions_in)
    combined = combined_body(shapes, np.array([-0., 0., 0.]), np.array([0., 0., 0.]))

    # set time
    time = 0
    total_time = 1.#.15#10

    # run the simulation
    run_sim(time, dt, total_time, shapes, combined, shape_shape_frictions,shape_ground_frictions, True, False, motion_script)

    print("Mass:",combined.mass)
    print("COM:",combined.COM)
    print("I:",combined.I)
    print("I for the y-axis:",combined.I[1][1])



#load shape info
print()
#combined_info = file_handling.read_combined_boxes_rigid_body_file("try1.txt")
combined_info = file_handling.read_combined_boxes_rigid_body_file("hammer.txt")
rotation = geometry_utils.normalize(np.array([0., 0.3, 0., 0.95]))

external_force_script = file_handling.read_external_force_script_file("external_force_script.csv")

# set dt
dt = 0.001


masses = np.linspace(1.5, 15., 50)
mu_values = np.linspace(0, 0.5, 50)

actual_mass_values = 1.*np.ones((len(combined_info)))
actual_mu_values = 0.3*np.ones((len(combined_info)))

##for 2-component shape:
#actual_mass_values[0]=10.
#for hammer
for i in np.arange(32):
    actual_mass_values[i] = 10.

#temp_mass_values = np.array([8.89699020224429, 14.9677209483574, 11.9923671466221, 7.15753688286746, 18.6387991789189, 3.48994516222535, 13.934303223965, 15.3254389078696, 5.87919201799735, 10.7787085222541, 12.7353274888345, 0.500483124956181, 9.96223770306502, 5.55420940004804, 5.76141497266729, 16.9575706700041, 15.4823478402692, 7.63684248836076, 7.56902605730114, 7.10143596403488, 9.80835991517899, 14.5938120473543, 7.51381919739439, 8.86454628705836, 15.5295813274777, 9.25839029269623, 0.54473080559343, 10.3495727088272, 9.79056314692446, 14.5566575477758, 8.7660572317503, 10.4181392812605, 12.813989526761, 6.47819701417439, 14.8021115286055, 7.74974085480524, 10.4387952461088, 12.3411597317592, 2.77220580044348, 15.0405411069867, 9.40156487414608, 7.64986923434638, 10.3101963220124, 13.2198344064352, 12.7439747096532, 13.6959556807137, 10.7200044914251, 3.50859391191897, 0.534520432064302, 12.7064582278817, 5.75745241135366, 8.45991547042282, 13.5642804514493, 6.94303371516847, 12.3634258089449, 2.22178204092344, 15.4290639163791,13.206444476902, 6.02631215478425, 10.8164080207691, 9.64131373952631, 0.552389968232394, 11.1076603001077, 6.41777469483705, 3.45824176354364, 0.567163256661271, 0.61778533988159, 5.26563277664322, 0.635826327450377, 0.749437027969597, 1.95088107753876, 0.500887223769284, 9.80297259603917, 10.356306738809, 0.615288038609257, 4.03877983721301, 13.1409802389594, 4.76855316353137, 12.2801243661453, 3.49729742681063, 10.5234018319807, 6.08589767062348, 6.15456963724876, 5.87947502173238, 0.633784241774706, 3.58591415182925, 12.7112126505709, 12.5305682137767])
#temp_friction_values = np.array([0.490472484091987, 0.46464007379533, 0.494486166837544, 0.488372376629397, 0.5, 0.49619382964266, 0.5, 0.490033197486532, 0.5, 0.495250930621612, 0.5, 0.485980444657035, 0.5, 0.494708652323586, 0.5, 0.48021700721466, 0.499538183322252, 0.491485576301457, 0.482610281230558, 0.487219102783097, 0.480912083380977, 0.466603695009185, 0.479814119608786, 0.471431326077002, 0.428281181847927, 0.449073698634622, 0.486480704447129, 0.0889062233423233, 0.470960689896326, 0.454310808052593, 0.469455102863066, 0.459462156338643, 0.445918172177983, 0.468097360373353, 0.422817131513697, 0.451983205457894, 0.433264635798613, 0.415335121328377, 0.479821706706827, 0.37928026877987, 0.423694321912463, 0.432548699451157, 0.411577097616278, 4.62849618901452E-05, 6.70635159169809E-05, 0.300117259465548, 0.323540480504563, 1.20481156084096E-05, 0.478952591959196, 1.7368248522866E-07, 5.29927984902358E-06, 6.41611190812451E-06, 3.85458636473809E-09, 7.3844723634914E-11, 2.32731586358741E-09, 1.65266192699108E-10, 0, 1.85365352729149E-06, 1.19508028597982E-06, 2.31216577732923E-06, 2.55359015425141E-06, 0, 0.000304859622388863, 0.000514156225708061, 0.000531468020041119, 0.0004193884378199, 0.00153133360111239, 0.00407712096563078, 0.0126935667734519, 0.000408890529404927, 0.00264140272397925, 0.00196745413461401, 0.0136504866525122, 0.0275576053686505, 0.000163503079092655, 0.0140496430571793, 0.0341380022245148, 0.0196357360455469, 0.0405068554688794, 0.0177223670111217, 0.0421099143733206, 0.0327269464467263, 0.0283679124804435, 0.0361797904756555, 0.00543332361040104, 0.0248391633462249, 0.0742208254615656, 0.0914493854054951])
#temp_mass_values = np.array([5.78391658015135, 4.7033595147844])
#temp_friction_values = np.array([0.5, 0.0555705990135459])
#temp_mass_values = np.concatenate((np.array([19.567496204797063]*32), np.array([7.092799154660835]*56)))
#temp_friction_values = np.concatenate((np.array([0.4381875344390944]*32), np.array([0.20276262173721]*56)))
#temp_mass_values = np.concatenate((np.array([7.4642306248406225]*32), np.array([3.535769377155203]*56)))
#temp_friction_values = np.concatenate((np.array([0.40191684167894487]*32), np.array([0.08484716205309362]*56)))
#temp_mass_values = np.concatenate((np.array([6.5]*32), np.array([3.]*56)))
#temp_friction_values = np.concatenate((np.array([3./6.5]*32), np.array([0.1]*56)))
temp_mass_values = np.concatenate((np.array([8.]*32), np.array([2.]*56)))

time_step = 100
#motion_script = file_handling.read_motion_script_file(os.path.join("test1","motion_script.csv"))

ordinary_run(actual_mass_values, actual_mu_values)
#ordinary_run(actual_mass_values, actual_mu_values, motion_script=motion_script)
#run_derivatives_sweep(1, False, masses, motion_script, time_step, temp_mass_values, temp_friction_values)
#run_derivatives_sweep(1, False, masses, motion_script, time_step, actual_mass_values, actual_mu_values)
#run_derivatives_sweep(0, True, mu_values, motion_script, time_step, actual_mass_values, actual_mu_values)
#run_2D_derivatives_sweep(0, 1, False, masses, motion_script, time_step, actual_mass_values, actual_mu_values)
#run_2D_derivatives_sweep(0, 1, True, mu_values, motion_script, time_step, actual_mass_values, actual_mu_values)

#run_2D_derivatives_sweep(11, 12, False, masses, motion_script, time_step, temp_mass_values, temp_friction_values)
#run_2D_derivatives_sweep(0, 1, False, masses, motion_script, time_step, temp_mass_values, temp_friction_values)
#run_2D_derivatives_sweep(0, 1, True, mu_values, motion_script, time_step, temp_mass_values, temp_friction_values)

'''loss_sweep_file = open("loss_sweep_file.csv", "w", encoding="utf-8")
loss_sweep_file.write("mass0,mass1,log(loss)\n")
run_2D_loss_sweep(0, 1, False, np.linspace(0.5, 11.5, 99), np.linspace(0.5, 11.5, 99), motion_script, time_step, np.array([actual_mass_values[0], actual_mass_values[1]]), np.array([temp_friction_values[0],temp_friction_values[60]]))
loss_sweep_file.close()'''


'''shapes_len = len(combined_info)
#initial_guess = np.concatenate((15*np.random.random(2)+0.5, (np.random.random(2))*0.45)) #for hammer, split into handle and head
initial_guess = np.concatenate((np.random.random(2), (np.random.random(2)))) #for hammer, split into handle and head, normalized
#initial_guess = np.concatenate((np.array([temp_mass_values[0],temp_mass_values[60]]), (np.random.random(2))*0.45)) #for hammer, split into handle and head
#initial_guess = np.concatenate((np.array([(temp_mass_values[0]-0.5)/19.5,(temp_mass_values[60]-0.5)/19.5]), (np.random.random(2))*0.45*2.)) #for hammer, split into handle and head, normalized
#initial_guess = np.concatenate((15*np.random.random(2)+0.5, np.array([temp_friction_values[0],temp_friction_values[60]]))) #for hammer, split into handle and head
#initial_guess = np.concatenate((np.array([(temp_mass_values[0]-0.5)/19.5,(temp_mass_values[60]-0.5)/19.5]), np.array([temp_friction_values[0]*2.,temp_friction_values[60]*2.]))) #for hammer, split into handle and head, normalized
#initial_guess = np.concatenate((np.array([actual_mass_values[0],actual_mass_values[60]]), np.array([actual_mu_values[0],actual_mu_values[60]]))) #for hammer, split into handle and head
#initial_guess = np.concatenate((np.array([(actual_mass_values[0]-0.5)/19.5,(actual_mass_values[60]-0.5)/19.5]), np.array([actual_mu_values[0]*2.,actual_mu_values[60]*2.]))) #for hammer, split into handle and head, normalized
#initial_guess = np.concatenate((15*np.random.random(2)+0.5, np.array([actual_mu_values[0],actual_mu_values[1]]))) #for hammer, split into handle and head

#initial_guess = np.concatenate((15*np.random.random(len(combined_info))+0.5, (np.random.random(len(combined_info)))*0.45))
#initial_guess = np.concatenate((15*np.random.random(len(combined_info))/19.5, (np.random.random(len(combined_info)))*0.45*2)) #normalized
#initial_guess = np.array([8.,3.,.375,.1])
#initial_guess = np.concatenate((15*np.random.random(len(combined_info))+0.5, 0.5*np.ones((len(combined_info)))))
#initial_guess = np.concatenate((actual_mass_values, (np.random.random(len(combined_info)))*0.45))
#initial_guess = np.concatenate((actual_mass_values, actual_mu_values))
#ordinary_run(initial_guess[:shapes_len], initial_guess[shapes_len:])
#ordinary_run(np.concatenate((np.repeat(376./32.*(.8*initial_guess[0]+.1), 32), np.repeat(376./56.*(1.-(.8*initial_guess[0]+.1)),56))), np.concatenate((np.repeat(initial_guess[2],32), np.repeat(initial_guess[3],56)))) #for hammer, split into handle and head
#bounds = [(0.5, 20.)]*len(combined_info) + [(0., 0.5)]*len(combined_info)
#bounds = [(0., 1.)]*len(combined_info) + [(0., 1.)]*len(combined_info) #normalized
#bounds = [(0.5, 20.)]*2 + [(0., 0.5)]*2 #for hammer, split into handle and head
bounds = [(0., 1.)]*2 + [(0., 1.)]*2 #for hammer, split into handle and head, normalized
print(initial_guess)
#print(initial_guess[0]*initial_guess[2],initial_guess[1]*initial_guess[3])
mult_factor = None
vals = find_values_L_BFSG_B(motion_script, time_step, initial_guess, bounds, shapes_len)
#ughudhrtuhg = input("Go?")
#ordinary_run(vals[:shapes_len], vals[shapes_len:])
ordinary_run(np.concatenate((np.repeat(376./32.*(.8*vals[0]+.1), 32), np.repeat(376./56.*(1.-(.8*vals[0]+.1)),56))), np.concatenate((np.repeat(vals[2]/2.,32), np.repeat(vals[3]/2.,56)))) #for hammer, split into handle and head
print(vals)
#print(vals[0]*vals[2],vals[1]*vals[3])'''

#run_mass_derivatives_sweep_in_combined_shape(0, masses, [1.,1.], [0.2, 0.02])
