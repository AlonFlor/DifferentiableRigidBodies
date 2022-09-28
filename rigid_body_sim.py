import numpy as np
import os
import contact_code
import geometry_utils
import file_handling


#rigid body classes
class shape:
    def __init__(self, shape_type, dimensions, location, orientation, velocity, angular_velocity, internal_points = None):
        self.shape_type = shape_type
        
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
        self.mass=1.
        self.center = np.array([0., 0., 0.])
        #add moment of inertia and center of mass
        self.I = np.zeros((3,3))
        self.COM = np.array([0.,0.,0.])
        if internal_points is not None:
            points = internal_points[:,0]
            masses = internal_points[:,1]
            self.mass=0
            for mass in masses:
                self.mass += mass
            for mass,point in zip(masses, points):
                self.I += mass*(np.square(point)*np.identity(3) - np.outer(point, point))
                self.COM += mass*point
        else:
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


class joint:
    def __init__(self, rotation_axis, shape, hinge_location):
        self.shape = shape
        self.angle = 0.
        self.screw_axis = np.append(rotation_axis, np.cross(-1*rotation_axis, hinge_location))



#make directory for simulation files
testNum = 1
while os.path.exists("test"+str(testNum)):
    testNum += 1
dir_name = "test"+str(testNum)
os.mkdir(dir_name)

#load and instantiate shapes
#box1 = shape("box", (1.,1.,1.), np.array([-3.0, 5.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([1., -1., 0.]), np.array([0.,np.pi/2,0.]))
#box1 = shape("box", (1.,1.,1.), np.array([0.0, 0, -5.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 2.]), np.array([0., 0., 0.]))
#box1 = shape("box", (1.,1.,1.), np.array([1.0, 0, -5.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 2.]), np.array([0., 0., 0.]))
#box1 = shape("box", (1.,1.,1.), np.array([0.0, -0.5, -5.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 2.]), np.array([0., 0., 0.]))
box1 = shape("box", (1.,1.,1.), np.array([1.0, -0.5, -5.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 2.]), np.array([0., 0., 0.]))
long_flat_box = shape("box", (7.,0.1,4.), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 0.]), np.array([0., 0., 0.]))
print()
shapes = []
shapes.append(box1)
shapes.append(long_flat_box)

#box1 = shape("box", (1.,1.,1.), np.array([0.0, 0, -5.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 2.]), np.array([0., 0., 0.]))
#rotate_box = geometry_utils.rotation_to_quaternion(np.pi/5, np.array([0., 1., 0.]))
#box1 = shape("box", (1.,1.,1.), np.array([0.0, 0, -5.0]), rotate_box, np.array([0., 0., 2.]), np.array([0., 0., 0.]))

rotate_cylinder = geometry_utils.rotation_to_quaternion(np.pi/2, np.array([0., 1., 0.]))
#cylinder1 = shape("uncapped cylinder", (.25,5.), np.array([0.0, 0.0, 0.0]), rotate_cylinder, np.array([0., 0., 0.]), np.array([0., 0., 0.]))
#cylinder2 = shape("uncapped cylinder", (.25,5.), np.array([0.0, 0, -5.0]), rotate_cylinder, np.array([0., 0.1, 2.]), np.array([0., 0., 0.]))
#rotate_cylinder2 = geometry_utils.rotation_to_quaternion(np.pi, np.array([0., 1., 0.]))
#cylinder3 = shape("uncapped cylinder", (.25,5.), np.array([0.0, 5.0, 0.0]), rotate_cylinder2, np.array([0., -1.0, 0.1]), np.array([0., 0., 0.]))

#sphere1 = shape("sphere", (.5), np.array([0.0, 0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 0.]), np.array([0., 0., 0.]))
#sphere2 = shape("sphere", (.5), np.array([0.0, 0.1, -5.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 2.]), np.array([0., 0., 0.]))

#cylinder4 = shape("uncapped cylinder", (.25,5.), np.array([0.0, 0.25, -5.0]), rotate_cylinder, np.array([0., 0., 2.]), np.array([0., 0., 0.]))

#shapes.append(box1)
#shapes.append(cylinder4)

'''shift = np.array([-12.5, 0., 0.])
arm_base = shape("box", (1.,1.,1.), np.array([0.0, 0.0, 0.0])+shift, np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 0.]), np.array([0., 0., 0.]))
shapes.append(arm_base)
arm_upper = shape("box", (1.,1.,5.), np.array([0.0, 0.0, 3.5])+shift, np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 0.]), np.array([0., 0., 0.]))
shapes.append(arm_upper)
arm_fore = shape("box", (1.,1.,5.), np.array([0.0, 0.0, 9.0])+shift, np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 0.]), np.array([0., 0., 0.]))
shapes.append(arm_fore)
arm_hand = shape("box", (1.,1.,1.), np.array([0.0, 0.0, 12.5])+shift, np.array([0.0, 0.0, 0.0, 1.0]), np.array([0., 0., 0.]), np.array([0., 0., 0.]))
shapes.append(arm_hand)'''

#create joints
joints = []
'''base_azimuth_rotate_joint = joint(np.array([0., 1., 0.]), arm_base, arm_base.location)
joints.append(base_azimuth_rotate_joint)
base_upper_altitude_joint = joint(np.array([-1., 0., 0.]), arm_upper, arm_base.location)
joints.append(base_upper_altitude_joint)
elbow_joint = joint(np.array([-1., 0., 0.]), arm_fore, arm_fore.location-np.array([0.,0.,2.75]))
joints.append(elbow_joint)
wrist_joint = joint(np.array([-1., 0., 0.]), arm_hand, arm_hand.location-np.array([0.,0.,0.75]))
joints.append(wrist_joint)'''

#create list of shape pairs that do not require collision testing since they are connected by a joint
fixed_contact_shapes = []
'''fixed_contact_shapes.append((arm_base, arm_upper))
fixed_contact_shapes.append((arm_upper, arm_fore))
fixed_contact_shapes.append((arm_fore, arm_hand))'''

original_locs_transformation_matrices = []
for i in np.arange(len(joints)):
    original_locs_transformation_matrices.append(geometry_utils.to_transformation_matrix(joints[i].shape.orientation, joints[i].shape.location))

def joints_reposition(angles):
    transformation_matrix = np.identity(4)
    for i in np.arange(len(joints)):
        joints[i].angle = angles[i] % (2*np.pi)
        #update transformation matrix
        transformation_matrix = np.matmul(transformation_matrix, geometry_utils.matrix_exponential(joints[i].screw_axis, joints[i].angle))
        #update local shape
        old_orientation = joints[i].shape.orientation
        old_location = joints[i].shape.location
        joints[i].shape.orientation, joints[i].shape.location = geometry_utils.from_transformation_matrix(np.matmul(transformation_matrix, original_locs_transformation_matrices[i]))
        joints[i].shape.velocity = (joints[i].shape.location - old_location)/dt
        joints[i].shape.angular_velocity = geometry_utils.angular_velocity(joints[i].shape.orientation, (joints[i].shape.orientation - old_orientation)/dt)


#set time
time = 0
dt = 0.001
total_time = 10

#set a script for the joints
joints_angles = [0., 0., 0., 0.]
#joint 0 from 0 to pi/2
joints_script0 = np.linspace(0, 1, int(2.5/dt)) * np.pi/2
#joint 1 from 0 to pi/4
joints_script1 = np.linspace(0, 1, int(2.5/dt)) * np.pi/4
#joint 2 from 0 to -pi/8
joints_script2_3 = np.linspace(0, 1, int(2.5/dt)) * -1*np.pi/8

#set up data storage
loc_file = os.path.join(dir_name, "data_locations.csv")
outfile = open(loc_file, "w")
outfile.write("time")
count = 1
for shape in shapes:
    outfile.write(",shape_"+str(count)+"_x")
    outfile.write(",shape_"+str(count)+"_y")
    outfile.write(",shape_"+str(count)+"_z")
    outfile.write(",shape_"+str(count)+"_quaternion_v1")
    outfile.write(",shape_"+str(count)+"_quaternion_v2")
    outfile.write(",shape_"+str(count)+"_quaternion_v3")
    outfile.write(",shape_"+str(count)+"_quaternion_s")
    count += 1
outfile.write("\n")


energies_file = os.path.join(dir_name, "data_energy.csv")
energies_outfile= open(energies_file, "w")
energies_outfile.write("time,KE,PE,total energy\n")
#momenta_file = os.path.join(dir_name, "data_momenta.csv")
#angular momentum file

step=0
while(time < total_time):
    if(step % 40 == 0):
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

    ##apply external forces
    #for shape in shapes:
    #    shape.velocity[1] -= 9.8*dt

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
    shape_shape_contacts = contact_code.shape_shape_collision_detection(collision_check_list)
    ground_contacts = []#no ground contacts   contact_code.shape_ground_collision_detection(ground_contacts_check_list)#  

    shape_shape_contact_impulses = []
    ground_contact_impulses = []
    for i in np.arange(len(shape_shape_contacts)):
        shape_shape_contact_impulses.append(np.array([0., 0., 0.]))
    for i in np.arange(len(ground_contacts)):
        ground_contact_impulses.append(np.array([0., 0., 0.]))

    #collision handling
    restitution = 1.
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
                impulse, r, I_inv = contact_code.shape_ground_collision_impulse(shape, contact, dv, restitution)
                contact_code.shape_ground_apply_impulse(shape, impulse, r, I_inv)

                #add impulse to record
                ground_contact_impulses[index] += impulse

            shape_shape_contacts_with_negative_dv = contact_code.shape_shape_contact_dv_check(shape_shape_contacts)
            shape_ground_contacts_with_negative_dv = contact_code.shape_ground_contact_dv_check(ground_contacts)

    #handle friction
    '''
        need pairwise tangential velocity
        does friction stop pairwise velocity before or after the contact?
        there should be no difference, since the collision impulse is perpendicular to friction
        there will most likely be a problem later, when I will have to distribute the impulses to preserve joint constraints,
        but for now just assume that all I need is the tangential velocity.

        so far this is kinetic friction. I will need another one for static friction.
        assume mu = 0.5 everywhere
    '''
    for i in np.arange(len(shape_shape_contacts)):
        normal_impulse_magn = np.linalg.norm(shape_shape_contact_impulses[i])
        if normal_impulse_magn <= 0.000001:     #threshold
            continue
        
        shape_pair, contact = shape_shape_contacts[i]
        shape1, shape2 = shape_pair
        world_point, normal = contact

        tangential_velocity = contact_code.get_shape_shape_tangential_velocity(shape1, shape2, world_point, normal)
        print("tangential_velocity",tangential_velocity)
        tangential_velocity_magn = np.linalg.norm(tangential_velocity)
        if tangential_velocity_magn <= 0.000001:     #threshold
            continue
        
        friction_direction = tangential_velocity / tangential_velocity_magn
        relative_motion_friction, r_1, r_2, I_inv_1, I_inv_2 = contact_code.shape_shape_collision_impulse(shape1, shape2, (world_point, friction_direction), tangential_velocity_magn, 0.)
        relative_motion_friction_magn = np.linalg.norm(relative_motion_friction)
        mu_normal_friction_magn = 0.5*normal_impulse_magn

        friction = friction_direction*min(relative_motion_friction_magn, mu_normal_friction_magn)
        print("friction",friction)
        contact_code.shape_shape_apply_impulse(shape1, shape2, friction, r_1, r_2, I_inv_1, I_inv_2)
    for i in np.arange(len(ground_contacts)):
        normal_impulse_magn = np.linalg.norm(ground_contact_impulses[i])
        if normal_impulse_magn <= 0.000001:     #threshold
            continue
        
        shape, contact = ground_contacts[i]
        world_point, normal = contact

        tangential_velocity = contact_code.get_shape_ground_tangential_velocity(shape, world_point, normal)
        print("tangential_velocity",tangential_velocity)
        tangential_velocity_magn = np.linalg.norm(tangential_velocity)
        if tangential_velocity_magn <= 0.000001:     #threshold
            continue
        
        friction_direction = tangential_velocity / tangential_velocity_magn
        relative_motion_friction, r, I_inv = contact_code.shape_ground_collision_impulse(shape, (world_point, friction_direction), tangential_velocity_magn, 0.)
        relative_motion_friction_magn = np.linalg.norm(relative_motion_friction)
        mu_normal_friction_magn = 0.5*normal_impulse_magn
        
        friction = friction_direction*min(relative_motion_friction_magn, mu_normal_friction_magn)
        print("friction",friction)
        contact_code.shape_ground_apply_impulse(shape, friction, r, I_inv)
        
    #run joints scripts
    '''if step < int(2.5/dt):
        joints_angles = [joints_script0[step], joints_angles[1], joints_angles[2], joints_angles[3]]
    elif step < int(5.0/dt):
        joints_angles = [joints_angles[0], joints_script1[step - int(2.5/dt)], joints_angles[2], joints_angles[3]]
    elif step < int(7.5/dt):
        joints_angles = [joints_angles[0], joints_angles[1], joints_script2_3[step - int(5.0/dt)], joints_angles[3]]
    else:
        joints_angles = [joints_angles[0], joints_angles[1], joints_angles[2], joints_script2_3[min(step - int(7.5/dt), 2499)]]
    joints_reposition(joints_angles)'''

    #update time
    #to do: make this symplectic (velocity verlet)
    time+=dt
    step+=1
    
outfile.close()

print("writing simulation files")
file_handling.write_simulation_files(shapes, loc_file, dir_name, dt, 24)
