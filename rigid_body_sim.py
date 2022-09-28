import numpy as np
import os
import contact_code
import geometry_utils
import file_handling
import draw_data


#rigid body classes
class rigid_shape:
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
                self.faces_sdf.append(contact_code.generate_signed_distance_function_for_face(self.center, self.vertices, self.face_normals, face))
            self.signed_distance = contact_code.generate_signed_distance_function_for_faces(self.faces_sdf)

        elif self.shape_type == "sphere":
            self.signed_distance = contact_code.generate_signed_distance_function_sphere(self.radius)
        elif self.shape_type == "uncapped cylinder":
            self.signed_distance = contact_code.generate_signed_distance_function_cylinder(self.radius)


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

        #set empty velocity derivatives and angular velocity derivatives with respect to mass and mu. Those are 0 initially
        self.velocity_mass_derivative = np.array([0., 0., 0.])
        self.angular_velocity_mass_derivative = np.array([0., 0., 0.])
        self.velocity_mu_derivative = np.array([0., 0., 0.])
        self.angular_velocity_mu_derivative = np.array([0., 0., 0.])

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
        for shape in self.components:
            self.mass += shape.mass
            world_COM += shape.mass*geometry_utils.to_world_coords(shape, shape.COM)
        self.location = world_COM / self.mass
        self.COM = np.array([0., 0., 0.])

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
        self.location_mass_derivatives = []
        #get mass derivatives of combined body's location, which is its center of mass in world coordinates
        second_term = world_COM / (self.mass * self.mass)
        for index in np.arange(len(self.components)):
            shape = self.components[index]
            self.location_mass_derivatives.append(geometry_utils.to_world_coords(shape, shape.COM) / self.mass - second_term)
            #d_sl/d_mass = d_world_COM/d_mass / self.mass - world_COM / self.mass / self.mass
        
    def set_component_velocities_and_angular_velocities(self):
        for i in np.arange(len(self.components)):
            shape = self.components[i]
            displacement = self.component_translations[i]
            displacement_w = geometry_utils.to_world_coords(self, displacement)
            shape_velocity_w = geometry_utils.velocity_of_point(self, displacement_w)
            shape.velocity = shape_velocity_w
            shape.angular_velocity = self.angular_velocity

    '''def set_component_locations_and_orientations(self):
        #this should only be called once, at the beginning of the simulation, to force the internal shapes to the location and orientation of the parent combined shape
        for i in np.arange(len(self.components)):
            shape = self.components[i]
            displacement = self.component_translations[i]
            shape.location = geometry_utils.to_world_coords(self, displacement)
            shape.orientation = geometry_utils.quaternion_mult(self.component_rotations[i], self.orientation)'''

def make_combined_boxes_rigid_body(info, location, velocity, orientation, angular_velocity, masses):
    internal_shapes = []
    center = np.array([0.,0.,0.])
    count = 0
    for coords in info:
        x,y,z = coords
        local_loc = np.array([x, y, z])
        center += local_loc
        internal_shapes.append(rigid_shape("box", (1.,1.,1.), local_loc, orientation, np.array([0., 0., 0.]), np.array([0., 0., 0.]), masses[count]))
        count += 1
    center /= len(info)
    for a_shape in internal_shapes:
        a_shape.location -= center
    for a_shape in internal_shapes:
        a_shape.location = geometry_utils.rotation_from_quaternion(orientation, a_shape.location) + location
    the_combined_body = combined_body(internal_shapes, velocity, angular_velocity)
    '''the_combined_body.location = location
    the_combined_body.orientation = orientation
    the_combined_body.set_component_locations_and_orientations()'''
    return the_combined_body





def run_sim(start_time, dt, total_time, shapes, combined, dir_name):
    time = start_time
    
    #set up data storage
    loc_file = os.path.join(dir_name, "data_locations.csv")
    outfile = open(loc_file, "w")
    outfile.write("time")
    count = 1
    for shape in shapes:
        outfile.write(",shape_"+str(count)+"_x")
        outfile.write(",shape_"+str(count)+"_y")
        outfile.write(",shape_"+str(count)+"_z")
        outfile.write(",shape_"+str(count)+"_quaternion_i")
        outfile.write(",shape_"+str(count)+"_quaternion_j")
        outfile.write(",shape_"+str(count)+"_quaternion_k")
        outfile.write(",shape_"+str(count)+"_quaternion_s")
        count += 1
    outfile.write("\n")


    energies_file = os.path.join(dir_name, "data_energy.csv")
    energies_outfile= open(energies_file, "w")
    energies_outfile.write("time,KE,PE,total energy\n")
    #momenta_file = os.path.join(dir_name, "data_momenta.csv")
    #angular momentum file

    combined_script = []
    find_derivatives = True

    step=0
    while(time < total_time):
        if True:#(step % 40 == 0):#
            print("simulation\tt =",time)
        #record
        outfile.write(str(time))
        energies_outfile.write(str(time))
        total_KE = 0
        total_PE = 0
        total_energy = 0
        for shape in shapes:
            total_KE += shape.KE
            total_PE += shape.PE
            total_energy += shape.KE + shape.PE
        energies_outfile.write(","+str(total_KE)+","+str(total_PE) + "," + str(total_energy) + "\n")
        for shape in shapes:
            for coord in shape.location:
                outfile.write(","+str(coord))
            for coord in shape.orientation:
                outfile.write(","+str(coord))
        outfile.write("\n")
        
        #move
        for shape in shapes:
            shape.location += dt*shape.velocity
            shape.orientation += dt*shape.orientation_derivative
            shape.orientation = geometry_utils.normalize(shape.orientation)
            shape.orientation_derivative = geometry_utils.orientation_derivative(shape.orientation, shape.angular_velocity)
            shape.KE = 0.5*shape.mass*np.square(np.linalg.norm(shape.velocity)) + 0.5*np.dot(np.matmul(shape.I,shape.angular_velocity), shape.angular_velocity)
            shape.PE = 0
        combined.location += dt*combined.velocity
        combined.orientation += dt*combined.orientation_derivative
        combined.orientation = geometry_utils.normalize(combined.orientation)
        combined.orientation_derivative = geometry_utils.orientation_derivative(combined.orientation, combined.angular_velocity)
        combined_script.append((step, combined.velocity, combined.angular_velocity))

        '''#apply external forces
        for shape in shapes:
            shape.velocity[1] -= 9.8*dt
        combined.velocity[1] -= 9.8*dt'''

        #update bounding boxes
        #bounding boxes consist of min_x,max_x,min_y,max_y,min_z,max_z
        for shape in shapes:
            if shape.shape_type=="box":
                world_vertices = []
                for vertex in shape.vertices:
                    world_vertices.append(geometry_utils.to_world_coords(shape, vertex))
                shape.bounding_box = geometry_utils.get_extrema(world_vertices)
            elif shape.shape_type=="sphere":
                shape.bounding_box = shape.location[0]-shape.radius,shape.location[0]+shape.radius, shape.location[1]-shape.radius,shape.location[1]+shape.radius, shape.location[2]-shape.radius,shape.location[2]+shape.radius
            elif shape.shape_type=="uncapped cylinder":
                shape.bounding_box = geometry_utils.get_cylinder_bounding_box_extrema(shape)
        

        #preliminary collision detection using bounding boxes
        collision_check_list = []
        ground_contacts_check_list = []
        for i in np.arange(len(shapes)):
            for j in np.arange(i+1, len(shapes)):
                if (shapes[i], shapes[j]) in fixed_contact_shapes:
                    continue
                if(contact_code.AABB_intersect(shapes[i],shapes[j])):
                    collision_check_list.append((shapes[i],shapes[j]))
            #check for ground collisions
            if shapes[i].bounding_box[2] < -0.00001:   #threshold
                ground_contacts_check_list.append(shapes[i])
        
        #main collision detection 
        shape_shape_contacts_low_level = []#no shape-shape collisions   contact_code.shape_shape_collision_detection(collision_check_list)
        ground_contacts_low_level = contact_code.shape_ground_collision_detection(ground_contacts_check_list)#  []#no ground contacts   

        #get friction coefficients
        shape_shape_contact_friction_coefficients = []
        ground_contact_friction_coefficients = []
        for i in np.arange(len(shape_shape_contacts_low_level)):
            shape_pair, contact = shape_shape_contacts_low_level[i]
            shape1, shape2 = shape_pair
            shape_shape_contact_friction_coefficients.append(shape_shape_frictions[shape1][shape2])
        for i in np.arange(len(ground_contacts_low_level)):
            shape, contact = ground_contacts_low_level[i]
            ground_contact_friction_coefficients.append(shape_ground_frictions[shape])

        '''#collision handling is done on the highest shape level, so get the parents of all shapes in contacts
        shape_shape_contacts = []
        ground_contacts = []
        for shape_pair, contact in shape_shape_contacts_low_level:
            shape1, shape2 = shape_pair
            new_shape1 = shape1
            new_shape2 = shape2
            if shape1.parent is not None:
                new_shape1 = shape1.parent
            if shape2.parent is not None:
                new_shape2 = shape2.parent
            shape_shape_contacts.append(((new_shape1, new_shape2), contact))
        for shape, contact in ground_contacts_low_level:
            new_shape = shape
            if shape.parent is not None:
                new_shape = shape.parent
            ground_contacts.append((new_shape, contact))'''
        ground_contacts = []
        for shape, contact in ground_contacts_low_level:
            ground_contacts.append((shape, contact))

        #empty impulse arrays to be filled during collision handling and used for calculating friction
        shape_shape_contact_impulses = []
        ground_contact_impulses = []
        '''for i in np.arange(len(shape_shape_contacts)):
            shape_shape_contact_impulses.append(np.array([0., 0., 0.]))'''
        for i in np.arange(len(ground_contacts)):
            ground_contact_impulses.append(np.array([0., 0., 0.]))
        if find_derivatives:
            ground_contact_impulses_mass_derivatives = []
            for i in np.arange(len(ground_contacts)):
                empty_array = []
                for shape in shapes:
                    empty_array.append(np.array([0., 0., 0.]))
                ground_contact_impulses_mass_derivatives.append(empty_array)

        #tangential velocity arrays calculated before impulses, for use in calculating friction
        shape_shape_contact_tangential_velocities = []
        ground_contact_tangential_velocities = []
        '''for i in np.arange(len(shape_shape_contacts)):
            shape_pair, contact = shape_shape_contacts[i]
            shape1, shape2 = shape_pair
            world_point, normal = contact
            tangential_velocity = contact_code.get_shape_shape_tangential_velocity(shape1, shape2, world_point, normal)
            shape_shape_contact_tangential_velocities.append(tangential_velocity)'''
        for i in np.arange(len(ground_contacts)):
            shape, contact = ground_contacts[i]
            world_point, normal = contact
            tangential_velocity = contact_code.get_shape_ground_tangential_velocity(shape, world_point, normal)
            ground_contact_tangential_velocities.append(tangential_velocity)

        '''#collision handling
        restitution = 0.
        if len(shape_shape_contacts) > 0 or len(ground_contacts) > 0:
            shape_shape_contacts_with_negative_dv = contact_code.shape_shape_contact_dv_check(shape_shape_contacts)
            shape_ground_contacts_with_negative_dv = contact_code.shape_ground_contact_dv_check(ground_contacts)
            while len(shape_shape_contacts_with_negative_dv) > 0 or len(shape_ground_contacts_with_negative_dv) > 0:
                for index, dv in shape_shape_contacts_with_negative_dv:
                    shape_pair, contact = shape_shape_contacts[index]
                    shape1, shape2 = shape_pair
                    impulse, r_1, r_2, I_inv_1, I_inv_2 = contact_code.shape_shape_collision_impulse(shape1, shape2, contact, dv, restitution)
                    contact_code.shape_shape_apply_impulse(shape1, shape2, impulse, r_1, r_2, I_inv_1, I_inv_2)

                    #add impulse to record
                    shape_shape_contact_impulses[index] += impulse
                    
                for index, dv in shape_ground_contacts_with_negative_dv:
                    shape, contact = ground_contacts[index]
                    if find_derivatives:
                        impulse, r, I_inv, impulse_mass_derivatives, r_mass_derivatives, I_inv_mass_derivatives = contact_code.shape_ground_collision_impulse(shape, contact, dv, restitution, True)
                        velocity_mass_derivatives, angular_velocity_mass_derivatives = contact_code.shape_ground_apply_impulse(shape, impulse, r, I_inv, impulse_mass_derivatives, I_inv_mass_derivatives)
                        for i in np.arange(len(shapes)):
                            shapes[i].velocity_mass_derivative += velocity_mass_derivatives[i]
                            shapes[i].angular_velocity_mass_derivative += angular_velocity_mass_derivatives[i]
                            ground_contact_impulses_mass_derivatives[index][i] += impulse_magn_mass_derivatives[i]#impulse_mass_derivatives[i]
                    else:
                        impulse, r, I_inv = contact_code.shape_ground_collision_impulse(shape, contact, dv, restitution, False)
                        contact_code.shape_ground_apply_impulse(shape, impulse, r, I_inv)

                    #add impulse to record
                    ground_contact_impulses[index] += dv#impulse_magn#impulse

                shape_shape_contacts_with_negative_dv = contact_code.shape_shape_contact_dv_check(shape_shape_contacts)
                shape_ground_contacts_with_negative_dv = contact_code.shape_ground_contact_dv_check(ground_contacts)'''

        #fill out impulse arrays
        for i in np.arange(len(ground_contacts)):
            ground_contact_impulses[i] = 9.8*shapes[i].mass*dt*np.array([0., 1., 0.])
            if find_derivatives:
                ground_contact_impulses_mass_derivatives[i][i] = 9.8*dt*np.array([0., 1., 0.])

        #handle friction
        '''
            need pairwise tangential velocity
            does friction stop pairwise velocity before or after the contact?
            there should be no difference, since the collision impulse is perpendicular to friction
            there will most likely be a problem later, when I will have to distribute the impulses to preserve joint constraints,
            but for now just assume that all I need is the tangential velocity.

            so far this is kinetic friction??? I will need another one for static friction???
        '''
        '''for i in np.arange(len(shape_shape_contacts)):
            normal_impulse_magn = np.linalg.norm(shape_shape_contact_impulses[i])
            if normal_impulse_magn <= 0.000001:     #threshold
                continue
            
            shape_pair, contact = shape_shape_contacts[i]
            shape1, shape2 = shape_pair
            world_point, normal = contact

            tangential_velocity = shape_shape_contact_tangential_velocities[i]
            print("tangential_velocity",tangential_velocity)
            tangential_velocity_magn = np.linalg.norm(tangential_velocity)
            if tangential_velocity_magn <= 0.000001:     #threshold
                continue

            mu = shape_shape_contact_friction_coefficients[i]
            
            friction_direction = tangential_velocity / tangential_velocity_magn
            relative_motion_friction, r_1, r_2, I_inv_1, I_inv_2 = contact_code.shape_shape_collision_impulse(shape1, shape2, (world_point, friction_direction), tangential_velocity_magn, 0.)
            relative_motion_friction_magn = np.linalg.norm(relative_motion_friction)
            mu_normal_friction_magn = mu*normal_impulse_magn

            friction = friction_direction*min(relative_motion_friction_magn, mu_normal_friction_magn)
            print("friction",friction)
            contact_code.shape_shape_apply_impulse(shape1, shape2, friction, r_1, r_2, I_inv_1, I_inv_2)'''
        for i in np.arange(len(ground_contacts)):
            normal_impulse_magn = np.linalg.norm(ground_contact_impulses[i])
            print("normal_impulse_magn",normal_impulse_magn)
            if normal_impulse_magn <= 0.000001:     #threshold
                continue

            if find_derivatives:
                normal_impulse_magn_mass_derivatives = []
                for j in np.arange(len(shapes)):
                    normal_impulse_magn_mass_derivatives.append(1. / normal_impulse_magn * np.linalg.norm(ground_contact_impulses[i] * ground_contact_impulses_mass_derivatives[i][j]))
                    print("normal_impulse_magn_mass_derivatives",normal_impulse_magn_mass_derivatives)
            
            shape, contact = ground_contacts_low_level[i]
            world_point, normal = contact

            tangential_velocity = ground_contact_tangential_velocities[i]
            print("tangential_velocity",tangential_velocity)
            tangential_velocity_magn = np.linalg.norm(tangential_velocity)
            if tangential_velocity_magn <= 0.000001:     #threshold
                continue

            mu = ground_contact_friction_coefficients[i]
            
            friction_direction = tangential_velocity / tangential_velocity_magn
            if find_derivatives:
                relative_motion_friction, r, I_inv, one_shape_friction_impulse_mass_derivatives, one_shape_friction_r_mass_derivatives, one_shape_friction_I_inv_mass_derivatives = contact_code.shape_ground_collision_impulse(shape, (world_point, friction_direction), tangential_velocity_magn, 0., True)
                friction_I_inv_mass_derivatives = []
                friction_impulse_mass_derivatives = []
                for j in np.arange(len(shapes)):
                    if shapes[j] is shape:
                        friction_I_inv_mass_derivatives.append(one_shape_friction_I_inv_mass_derivatives[0])
                        friction_impulse_mass_derivatives.append(one_shape_friction_impulse_mass_derivatives[0])
                    else:
                        friction_I_inv_mass_derivatives.append(np.array([0., 0., 0.]))
                        friction_impulse_mass_derivatives.append(np.array([0., 0., 0.]))
                relative_motion_friction_magn = np.linalg.norm(relative_motion_friction)
                relative_motion_friction_magn_mass_derivatives = []
                for j in np.arange(len(shapes)):
                    relative_motion_friction_magn_mass_derivatives.append(1. / relative_motion_friction_magn * relative_motion_friction * friction_impulse_mass_derivatives[j])
                mu_normal_friction_magn = mu*normal_impulse_magn
                mu_normal_friction_magn_mass_derivatives = []
                mu_normal_friction_magn_mu_derivative = []
                for j in np.arange(len(shapes)):
                    mu_normal_friction_magn_mass_derivatives.append(normal_impulse_magn_mass_derivatives[j] * mu)
                mu_normal_friction_magn_mu_derivative = normal_impulse_magn
                friction = friction_direction*min(relative_motion_friction_magn, mu_normal_friction_magn)
                rm = relative_motion_friction_magn < mu_normal_friction_magn
                print("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",rm)
                print("\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",relative_motion_friction_magn - mu_normal_friction_magn)
                friction_mass_derivatives = []
                if rm:
                    for j in np.arange(len(shapes)):
                        friction_mass_derivatives.append(relative_motion_friction_magn_mass_derivatives[j] * friction_direction)
                    friction_mu_derivative = 0 * friction_direction
                else:
                    #includes chink where relative_motion_friction_magn is equal to mu_normal_friction_magn
                    for j in np.arange(len(shapes)):
                        friction_mass_derivatives.append(mu_normal_friction_magn_mass_derivatives[j] * friction_direction)
                    friction_mu_derivative = mu_normal_friction_magn_mu_derivative * friction_direction
                if i==0:
                    friction0 = friction
                    friction_mu_derivatives0 = friction_mu_derivative
            else:
                relative_motion_friction, r, I_inv = contact_code.shape_ground_collision_impulse(shape, (world_point, friction_direction), tangential_velocity_magn, 0., False)
                relative_motion_friction_magn = np.linalg.norm(relative_motion_friction)
                mu_normal_friction_magn = mu*normal_impulse_magn
                friction = friction_direction*min(relative_motion_friction_magn, mu_normal_friction_magn)
                
            print("friction",friction)
            if shape.parent is not None:
                r = world_point - shape.parent.COM - shape.parent.location
                R = geometry_utils.quaternion_to_rotation_matrix(shape.parent.orientation)
                I_inv = np.matmul(R,np.matmul(shape.parent.I_inv,R.T))
                if find_derivatives:
                    r_mass_derivatives = []
                    I_inv_mass_derivatives = []
                    for j in np.arange(len(shapes)):
                        I_inv_mass_derivatives.append(np.matmul(R,np.matmul(shape.parent.I_inv_mass_derivatives[j],R.T)))
                        r_mass_derivatives.append(-1*shape.parent.location_mass_derivatives[i])
                    velocity_mass_derivatives, angular_velocity_mass_derivatives, velocity_mu_derivative, angular_velocity_mu_derivative = contact_code.shape_ground_apply_impulse(shape.parent, friction, r, I_inv, shapes.index(shape), r_mass_derivatives, friction_mass_derivatives, I_inv_mass_derivatives, friction_mu_derivative)
                    for j in np.arange(len(shapes)):
                        shapes[j].velocity_mass_derivative += velocity_mass_derivatives[j]
                        shapes[j].angular_velocity_mass_derivative += angular_velocity_mass_derivatives[j]
                    shape.velocity_mu_derivative += velocity_mu_derivative
                    shape.angular_velocity_mu_derivative += angular_velocity_mu_derivative
                else:
                    contact_code.shape_ground_apply_impulse(shape.parent, friction, r, I_inv)
            '''else:
                if find_derivatives:
                    velocity_mass_derivatives, angular_velocity_mass_derivatives = contact_code.shape_ground_apply_impulse(shape.parent, friction, r, I_inv, friction_mass_derivatives, friction_I_inv_mass_derivatives)
                    for j in np.arange(len(shapes)):
                            shapes[j].velocity_mass_derivative += velocity_mass_derivatives[j]
                            shapes[j].angular_velocity_mass_derivative += angular_velocity_mass_derivatives[j]
                else:
                    contact_code.shape_ground_apply_impulse(shape, friction, r, I_inv)'''
        

        #set velocities and angular velocities
        combined.set_component_velocities_and_angular_velocities()

        #update time
        #to do: make this symplectic (velocity verlet)
        time+=dt
        step+=1
        
    outfile.close()

    return shapes[0].velocity[2], shapes[0].velocity_mu_derivative[2]#friction0[2], friction_mu_derivatives0[2]#


#set time
time = 0
dt = 0.001
total_time = 0.002#10

#make directory for simulation files
testNum = 1
while os.path.exists("test"+str(testNum)):
    testNum += 1
dir_name = "test"+str(testNum)
os.mkdir(dir_name)

#load and instantiate shapes
print()
combined_info = file_handling.read_combined_boxes_rigid_body_file("try1.txt")
rotation = geometry_utils.normalize(np.array([0., 0.3, 0., 0.95]))
###
masses = np.linspace(0.5, 5., 1000)
mu_values = np.linspace(0, 0.5, 1000)
result = []
result_deriv = []
result_deriv_estimate = []
indices = (1,1)

outermost_count = 0
#for mass in masses:
for mu in mu_values:
    component_masses = [1., 1.]
    combined = make_combined_boxes_rigid_body(combined_info, np.array([0., 0.4999804, 5.]), np.array([0., 0., -1.]), rotation, np.array([0., 0., 0.]), component_masses)
    #print(combined.components[0].location)

    '''result.append(combined.location[2])#(combined.I_inv[indices[0]][indices[1]])
    result_deriv.append(combined.location_mass_derivatives[0][2])#(combined.I_inv_mass_derivatives[0][indices[0]][indices[1]])
    
    if outermost_count != 0:
        result_deriv_estimate.append((result[outermost_count]-result[outermost_count - 1]) / (mass - masses[outermost_count - 1]))
    else:
        result_deriv_estimate.append(0)
    outermost_count += 1'''
###
    shapes = []
    shapes += combined.components

    #set coefficients of friction for all shapes
    shape_shape_frictions = {}
    count = 0
    for shape in shapes:
        this_shape_frictions = {}
        for shape2 in shapes[count+1:]:
            this_shape_frictions[shape2] = 0.5          #set friction coefficient to 0.5 for now
        shape_shape_frictions[shape] = this_shape_frictions
        count += 1
    shape_ground_frictions = {}
    #for shape in shapes:
    #    shape_ground_frictions[shape] = 0.02             #set friction coefficient to 0.02 for now
    shape_ground_frictions[shapes[0]] = mu
    shape_ground_frictions[shapes[1]] = 0.02
        

    #create list of shape pairs that do not require collision testing since they are connected by a joint
    fixed_contact_shapes = []
    for i in np.arange(len(combined.components)):
        for j in np.arange(i+1,len(combined.components)):
            fixed_contact_shapes.append((combined.components[i], combined.components[j]))

    #run the simulation
    res1, res1_mass_deriv = run_sim(time, dt, total_time, shapes, combined, dir_name)

    #result.append(shapes[0].velocity[1])
    #result_deriv.append(shapes[0].velocity_mass_derivative[1])
    result.append(res1)
    result_deriv.append(res1_mass_deriv)
    
    if outermost_count != 0:
        result_deriv_estimate.append((result[outermost_count]-result[outermost_count - 1]) / (mu - mu_values[outermost_count - 1]))#(mass - masses[outermost_count - 1]))
    else:
        result_deriv_estimate.append(0)
    outermost_count += 1
    
draw_data.plot_data(mu_values, result, result_deriv, result_deriv_estimate)
for i in range(len(masses)):
    print(result_deriv_estimate[i] - result_deriv[i])
exit()

print("writing simulation files")
file_handling.write_simulation_files(shapes, loc_file, dir_name, dt, 24)
