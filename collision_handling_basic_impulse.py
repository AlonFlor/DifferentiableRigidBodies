import numpy as np
import geometry_utils



def shape_shape_contact_dv_check(contacts):
    #the normal is coming out of shape2
    contacts_with_negative_dv = []
    count = 0
    for shape_pair,contact in contacts:
        shape1, shape2 = shape_pair
        this_contact, normal = contact
        print("this_contact", this_contact, "normal", normal)
        shape1_point_v = geometry_utils.velocity_of_point(shape1, this_contact)
        shape2_point_v = geometry_utils.velocity_of_point(shape2, this_contact)
        print("shape1_point_v",shape1_point_v)
        print("shape2_point_v",shape2_point_v)
        dv = np.dot(normal, shape1_point_v - shape2_point_v)
        print("dv",dv)
        if dv < -0.00001:   #threshold
            contacts_with_negative_dv.append((count,dv))
        count += 1
    return contacts_with_negative_dv

#not differentiable
def shape_shape_collision_impulse(shape1, shape2, contact, dv, restitution):
    #the normal is coming out of shape2
    world_vertex, normal = contact
    
    #p_1 = geometry_utils.to_local_coords(shape1, world_vertex)
    #p_2 = geometry_utils.to_local_coords(shape2, world_vertex)
    r_1 = world_vertex - geometry_utils.to_world_coords(shape1,shape1.COM)
    r_2 = world_vertex - geometry_utils.to_world_coords(shape2,shape2.COM)

    R_1 = geometry_utils.quaternion_to_rotation_matrix(shape1.orientation)
    R_2 = geometry_utils.quaternion_to_rotation_matrix(shape2.orientation)
    I_inv_1 = np.matmul(R_1,np.matmul(shape1.I_inv,R_1.T))
    I_inv_2 = np.matmul(R_2,np.matmul(shape2.I_inv,R_2.T))
    
    rotational_part_1 = np.dot(normal, np.cross(np.matmul(I_inv_1, np.cross(r_1,normal)), r_1))
    rotational_part_2 = np.dot(normal, np.cross(np.matmul(I_inv_2, np.cross(r_2,normal)), r_2))

    print("normal:",normal)
    
    impulse_magn = (1. + restitution)*dv
    impulse_denom = 1./shape1.mass + 1./shape2.mass + rotational_part_1 + rotational_part_2
    impulse_magn = impulse_magn / impulse_denom

    impulse = impulse_magn*normal

    return impulse, r_1, r_2, I_inv_1, I_inv_2

#not differentiable
def shape_shape_apply_impulse(shape1, shape2, impulse, r_1, r_2, I_inv_1, I_inv_2):
    shape1.velocity -= impulse / shape1.mass
    shape2.velocity += impulse / shape2.mass

    shape1.angular_velocity -= np.matmul(I_inv_1,np.cross(r_1,impulse))
    shape2.angular_velocity += np.matmul(I_inv_2,np.cross(r_2,impulse))

    
    print("shape1.velocity -= ",impulse / shape1.mass)
    print("shape2.velocity += ",impulse / shape2.mass)

    print("shape1.angular_velocity -= ",np.matmul(I_inv_1,np.cross(r_1,impulse)))
    print("shape2.angular_velocity += ",np.matmul(I_inv_2,np.cross(r_2,impulse)))

    return




def shape_ground_contact_dv_check(contacts):
    #normal is coming out of the ground
    contacts_with_negative_dv = []
    count = 0
    for shape, contact_info in contacts:
        print("ground contact")
        contact_location, normal = contact_info
        shape_point_v = geometry_utils.velocity_of_point(shape, contact_location)
        dv = np.dot(normal, shape_point_v)
        print("dv",dv)
        if dv < -0.00001:   #threshold
            contacts_with_negative_dv.append((count,dv))
        count += 1
    return contacts_with_negative_dv

def shape_ground_collision_impulse(shape, contact, dv, restitution):
    world_vertex, normal = contact
    
    #p = geometry_utils.to_local_coords(shape, world_vertex)
    r = world_vertex - geometry_utils.to_world_coords(shape, shape.COM)

    R = geometry_utils.quaternion_to_rotation_matrix(shape.orientation)
    I_inv = np.matmul(R,np.matmul(shape.I_inv,R.T))
    
    rotational_part = np.dot(normal, np.cross(np.matmul(I_inv, np.cross(r,normal)), r))

    impulse_num = (1. + restitution)*dv
    impulse_denom = 1./shape.mass + rotational_part
    impulse_magn = impulse_num / impulse_denom


    impulse = impulse_magn*normal

    return impulse, r, I_inv, R, impulse_denom, impulse_num

def shape_ground_collision_impulse_derivatives(shape, contact, r, R, I_inv, impulse_denom, impulse_num):
    world_vertex, normal = contact

    # derivatives
    I_inv_mass_derivatives = []
    rotational_part_mass_derivatives = []
    r_cross_normal = np.cross(r, normal)
    for I_inv_mass_derivative in shape.I_inv_mass_derivatives:
        rotated_I_inv_mass_derivative = np.matmul(R, np.matmul(I_inv_mass_derivative, R.T))
        I_inv_mass_derivatives.append(rotated_I_inv_mass_derivative)

        rotational_part_mass_derivatives.append(np.dot(normal, np.cross(np.matmul(rotated_I_inv_mass_derivative, r_cross_normal), r)))
    #if shape.COM_mass_derivatives is not None:
    #    I_inv = np.matmul(R, np.matmul(shape.I_inv, R.T))
    #    for COM_mass_derivative in shape.COM_mass_derivatives:
    #        rotational_part_mass_derivatives.append(np.dot(normal, np.cross(np.matmul(I_inv, r_cross_normal), -COM_mass_derivative)))
    #        rotational_part_mass_derivatives.append(np.dot(normal, np.cross(np.matmul(I_inv, np.cross(-COM_mass_derivative, normal)), r)))

    mass_inv_sq = -1./(shape.mass*shape.mass)
    impulse_denom_mass_derivatives = []
    for rotational_part_mass_derivative in rotational_part_mass_derivatives:
        impulse_denom_mass_derivatives.append(mass_inv_sq + rotational_part_mass_derivative)

    impulse_magn_mass_derivatives = []
    impulse_magn_const_part = -1 * impulse_num / (impulse_denom * impulse_denom)
    for impulse_denom_mass_derivative in impulse_denom_mass_derivatives:
        impulse_magn_mass_derivatives.append(impulse_magn_const_part * impulse_denom_mass_derivative)

    impulse_mass_derivatives = []
    for impulse_magn_mass_derivative in impulse_magn_mass_derivatives:
        impulse_mass_derivatives.append(impulse_magn_mass_derivative * normal)

    return impulse_mass_derivatives, I_inv_mass_derivatives

def shape_ground_apply_impulse(shape, impulse, r, I_inv):
    #since we are dealing with the object on the ground here, only y-axis changes can be made to angular velocity
    shape.velocity += impulse / shape.mass
    shape.angular_velocity += np.matmul(I_inv,np.cross(r,impulse))*np.array([0.,1.,0.])

    #print("shape.velocity -= ",impulse / shape.mass)
    #print("shape.angular_velocity -= ",np.matmul(I_inv,np.cross(r,impulse))*np.array([0.,1.,0.]))

def shape_ground_apply_impulse_derivatives(shape, impulse, r, I_inv, component_index, impulse_mass_derivatives, I_inv_mass_derivatives, impulse_mu_derivative=None):
    #derivatives
    mass_inv = 1/shape.mass
    velocity_mass_derivatives = []
    angular_velocity_mass_derivatives = []

    for i in np.arange(len(impulse_mass_derivatives)):
        impulse_mass_derivative = impulse_mass_derivatives[i]
        I_inv_mass_derivative = I_inv_mass_derivatives[i]
        velocity_mass_derivatives.append(mass_inv * impulse_mass_derivative - impulse * mass_inv*mass_inv)
        angular_velocity_mass_derivatives.append(np.array([0.,1.,0.])*(np.matmul(I_inv,np.cross(r,impulse_mass_derivative)) + np.matmul(I_inv_mass_derivative,np.cross(r,impulse))))
    #print("velocity mass derivatives from this force:",velocity_mass_derivatives)
    #print("angular velocity mass derivatives from this force:", angular_velocity_mass_derivatives)
    '''total_v = 0.
    total_a = 0.
    for i in np.arange(len(impulse_mass_derivatives)):
        total_v+=velocity_mass_derivatives[i]
        total_a+=angular_velocity_mass_derivatives[i]
    print("subtotals delta mass derivatives from this force:",total_v[0], total_v[2], total_a[1])
    print("total delta mass derivatives from this force:",total_v[0] + total_v[2] + total_a[1])'''

    if impulse_mu_derivative is not None:
        velocity_mu_derivative = -1*impulse_mu_derivative / shape.mass
        angular_velocity_mu_derivative = -1*np.matmul(I_inv,np.cross(r,impulse_mu_derivative))*np.array([0.,1.,0.])
        #print("velocity mu derivative:",velocity_mu_derivative)
        #print("angular velocity mu derivative:",angular_velocity_mu_derivative)

    #add the mass and friction derivatives to the shape's records
    for i in np.arange(len(impulse_mass_derivatives)):
        shape.velocity_mass_derivatives[i] += velocity_mass_derivatives[i]
        shape.angular_velocity_mass_derivatives[i] += angular_velocity_mass_derivatives[i]
    if component_index is not None:
        shape.velocity_mu_derivatives[component_index] += velocity_mu_derivative
        shape.angular_velocity_mu_derivatives[component_index] += angular_velocity_mu_derivative
    


def get_shape_shape_tangential_velocity(shape1, shape2, contact_location, normal):
    print("normal",normal)
    shape1_velocity = geometry_utils.velocity_of_point(shape1, contact_location)
    shape2_velocity = geometry_utils.velocity_of_point(shape2, contact_location)
    print("shape1_velocity",shape1_velocity)
    print("shape2_velocity",shape2_velocity)

    shape1_velocity_normal = normal*np.dot(shape1_velocity, normal)
    shape2_velocity_normal = normal*np.dot(shape2_velocity, normal)
    print("shape1_velocity_normal",shape1_velocity_normal)
    print("shape2_velocity_normal",shape2_velocity_normal)

    shape1_velocity_perpendicular = shape1_velocity - shape1_velocity_normal
    shape2_velocity_perpendicular = shape2_velocity - shape2_velocity_normal
    print("shape1_velocity_perpendicular",shape1_velocity_perpendicular)
    print("shape2_velocity_perpendicular",shape2_velocity_perpendicular)

    return shape1_velocity_perpendicular - shape2_velocity_perpendicular

def get_shape_ground_tangential_velocity(shape, contact_location, normal):
    shape_velocity = geometry_utils.velocity_of_point(shape, contact_location)

    shape_velocity_normal = normal*np.dot(shape_velocity, normal)

    shape_velocity_perpendicular_to_normal = shape_velocity - shape_velocity_normal

    return shape_velocity_perpendicular_to_normal


def handle_collisions_using_impulses(shapes, ground_contacts_low_level, find_derivatives, dt, ground_contact_friction_coefficients):
    #shape-shape collisions are not included

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

    # empty impulse arrays to be filled during collision handling and used for calculating friction
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

    # tangential velocity arrays calculated before impulses, for use in calculating friction
    shape_shape_contact_tangential_velocities = []
    ground_contact_tangential_velocities = []
    '''for i in np.arange(len(shape_shape_contacts)):
        shape_pair, contact = shape_shape_contacts[i]
        shape1, shape2 = shape_pair
        world_point, normal = contact
        tangential_velocity = get_shape_shape_tangential_velocity(shape1, shape2, world_point, normal)
        shape_shape_contact_tangential_velocities.append(tangential_velocity)'''
    for i in np.arange(len(ground_contacts)):
        shape, contact = ground_contacts[i]
        world_point, normal = contact
        tangential_velocity = get_shape_ground_tangential_velocity(shape, world_point, normal)
        ground_contact_tangential_velocities.append(tangential_velocity)

    '''#collision handling
    restitution = 0.
    if len(shape_shape_contacts) > 0 or len(ground_contacts) > 0:
        shape_shape_contacts_with_negative_dv = shape_shape_contact_dv_check(shape_shape_contacts)
        shape_ground_contacts_with_negative_dv = shape_ground_contact_dv_check(ground_contacts)
        while len(shape_shape_contacts_with_negative_dv) > 0 or len(shape_ground_contacts_with_negative_dv) > 0:
            for index, dv in shape_shape_contacts_with_negative_dv:
                shape_pair, contact = shape_shape_contacts[index]
                shape1, shape2 = shape_pair
                impulse, r_1, r_2, I_inv_1, I_inv_2 = shape_shape_collision_impulse(shape1, shape2, contact, dv, restitution)
                shape_shape_apply_impulse(shape1, shape2, impulse, r_1, r_2, I_inv_1, I_inv_2)
    
                #add impulse to record
                shape_shape_contact_impulses[index] += impulse
    
            for index, dv in shape_ground_contacts_with_negative_dv:
                shape, contact = ground_contacts[index]
                impulse, r, I_inv, R, impulse_denom, impulse_num = shape_ground_collision_impulse(shape, contact, dv, restitution)
                shape_ground_apply_impulse(shape, impulse, r, I_inv)
                if find_derivatives:
                    impulse_mass_derivatives, I_inv_mass_derivatives = shape_ground_collision_impulse_derivatives(shape, contact, r, R, I_inv, impulse_denom, impulse_num)
                    shape_ground_apply_impulse_derivatives(shape, impulse, r, I_inv, component_index, impulse_mass_derivatives, I_inv_mass_derivatives)
                    #component_index is due to assumption of impulses being applied to the components of the combined shape. This is important to consider if/when I uncomment this ground impulse code.
                    
                #add impulse to record
                ground_contact_impulses[index] += impulse
    
            shape_shape_contacts_with_negative_dv = shape_shape_contact_dv_check(shape_shape_contacts)
            shape_ground_contacts_with_negative_dv = shape_ground_contact_dv_check(ground_contacts)'''

    # fill out ground impulse arrays, substituting for above commented out codeto calculate ground impulses explicitly
    for i in np.arange(len(ground_contacts)):
        ground_contact_impulses[i] = 9.8 * shapes[i].mass * dt * np.array([0., 1., 0.])
        if find_derivatives:
            ground_contact_impulses_mass_derivatives[i][i] = 9.8 * dt * np.array([0., 1., 0.])

    # handle friction
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
        relative_motion_friction, r_1, r_2, I_inv_1, I_inv_2 = shape_shape_collision_impulse(shape1, shape2, (world_point, friction_direction), tangential_velocity_magn, 0.)
        relative_motion_friction_magn = np.linalg.norm(relative_motion_friction)
        mu_normal_friction_magn = mu*normal_impulse_magn
    
        friction = friction_direction*min(relative_motion_friction_magn, mu_normal_friction_magn)
        print("friction",friction)
        shape_shape_apply_impulse(shape1, shape2, friction, r_1, r_2, I_inv_1, I_inv_2)'''
    total_friction = np.array([0., 0., 0.])
    for i in np.arange(len(ground_contacts)):
        normal_impulse_magn = np.linalg.norm(ground_contact_impulses[i])
        #print("normal_impulse_magn", normal_impulse_magn)
        if normal_impulse_magn <= 0.000001:  # threshold
            continue

        shape, contact = ground_contacts_low_level[i]
        world_point, normal = contact

        tangential_velocity = ground_contact_tangential_velocities[i]
        #print("tangential_velocity", tangential_velocity)
        tangential_velocity_magn = np.linalg.norm(tangential_velocity)
        if tangential_velocity_magn <= 0.000001:  # threshold
            continue

        mu = ground_contact_friction_coefficients[i]
        restitution = 0.

        friction_direction = -1*tangential_velocity / tangential_velocity_magn
        relative_motion_friction, r, I_inv, R, impulse_denom, impulse_num = shape_ground_collision_impulse(shape, (world_point, friction_direction), tangential_velocity_magn, restitution)
        relative_motion_friction_magn = np.linalg.norm(relative_motion_friction)
        mu_normal_friction_magn = mu * normal_impulse_magn
        friction = friction_direction * min(relative_motion_friction_magn, mu_normal_friction_magn)
        if(relative_motion_friction_magn < mu_normal_friction_magn):
            print("got it at shape ", i)
        if shape.parent is not None:
            r_combined = world_point - shape.parent.location
            R_combined = geometry_utils.quaternion_to_rotation_matrix(shape.parent.orientation)
            I_inv_combined = np.matmul(R_combined, np.matmul(shape.parent.I_inv, R_combined.T))
            shape_ground_apply_impulse(shape.parent, friction, r_combined, I_inv_combined)
        else:
            shape_ground_apply_impulse(shape, friction, r, I_inv)
        total_friction+=friction

        if find_derivatives:
            normal_impulse_magn_mass_derivatives = []
            for j in np.arange(len(shapes)):
                normal_impulse_magn_mass_derivatives.append(1. / normal_impulse_magn * np.dot(ground_contact_impulses[i], ground_contact_impulses_mass_derivatives[i][j]))
                one_shape_friction_impulse_mass_derivatives, one_shape_friction_I_inv_mass_derivatives = shape_ground_collision_impulse_derivatives(shape, (world_point, friction_direction), r, R, I_inv, impulse_denom, impulse_num)

            friction_I_inv_mass_derivatives = []
            friction_impulse_mass_derivatives = []
            for j in np.arange(len(shapes)):
                if shapes[j] is shape:
                    friction_I_inv_mass_derivatives.append(one_shape_friction_I_inv_mass_derivatives[0])
                    friction_impulse_mass_derivatives.append(one_shape_friction_impulse_mass_derivatives[0])
                else:
                    friction_I_inv_mass_derivatives.append(np.array([0., 0., 0.]))
                    friction_impulse_mass_derivatives.append(np.array([0., 0., 0.]))
            relative_motion_friction_magn_mass_derivatives = []
            for j in np.arange(len(shapes)):
                relative_motion_friction_magn_mass_derivatives.append(1. / relative_motion_friction_magn * np.dot(relative_motion_friction, friction_impulse_mass_derivatives[j]))
            mu_normal_friction_magn_mass_derivatives = []
            for j in np.arange(len(shapes)):
                mu_normal_friction_magn_mass_derivatives.append(normal_impulse_magn_mass_derivatives[j] * mu)
            mu_normal_friction_magn_mu_derivative = normal_impulse_magn
            rm = relative_motion_friction_magn < mu_normal_friction_magn
            friction_mass_derivatives = []
            if rm:
                for j in np.arange(len(shapes)):
                    friction_mass_derivatives.append(relative_motion_friction_magn_mass_derivatives[j] * friction_direction)
                friction_mu_derivative = 0 * friction_direction
            else:
                # includes chink where relative_motion_friction_magn is equal to mu_normal_friction_magn
                for j in np.arange(len(shapes)):
                    friction_mass_derivatives.append(mu_normal_friction_magn_mass_derivatives[j] * friction_direction)
                friction_mu_derivative = mu_normal_friction_magn_mu_derivative * friction_direction

            #print("friction", friction)

            if shape.parent is not None:
                I_inv_combined_mass_derivatives = []
                for j in np.arange(len(shapes)):
                    I_inv_combined_mass_derivatives.append(np.matmul(R_combined, np.matmul(shape.parent.I_inv_mass_derivatives[j], R_combined.T)))
                shape_ground_apply_impulse_derivatives(shape.parent, friction, r_combined, I_inv_combined, shapes.index(shape), friction_mass_derivatives, I_inv_combined_mass_derivatives, friction_mu_derivative)

            else:
                shape_ground_apply_impulse_derivatives(shape, friction, r, I_inv, None, friction_mass_derivatives, one_shape_friction_I_inv_mass_derivatives, friction_mu_derivative)

    print("total_friction",total_friction)
    return total_friction

def external_force_impulse(combined, external_force_magn, component_number, direction, dt, find_derivatives):
    # add external force collision
    '''
    # get first vertex on the first collision shape with the smallest x coord
    current_x = combined.components[0].vertices[0][0]
    vertex_index = 0
    for i, vertex in enumerate(combined.components[0].vertices):
        new_x = vertex[0]
        if new_x < current_x:
            current_x = new_x
            vertex_index = i
    '''
    #get first vertex on the first collision shape
    # contact is at the center of mass along the y-axis (height)
    external_force_contact_location = geometry_utils.to_world_coords(combined.components[component_number], combined.components[component_number].vertices[0]) * np.array([1., 0., 1.])
    external_force_contact_location[1] = combined.components[component_number].location[1]
    external_force_direction = np.array(direction)
    external_force_impulse_magn = external_force_magn * dt
    external_force_impulse = external_force_impulse_magn*external_force_direction

    r = external_force_contact_location - geometry_utils.to_world_coords(combined, combined.COM)
    R = geometry_utils.quaternion_to_rotation_matrix(combined.orientation)
    I_inv = np.matmul(R, np.matmul(combined.I_inv, R.T))

    shape_ground_apply_impulse(combined, external_force_impulse, r, I_inv)

    if find_derivatives:
        external_force_impulse_mass_derivatives = []
        I_inv_mass_derivatives = []

        for i in np.arange(len(combined.components)):
            I_inv_mass_derivatives.append(np.matmul(R, np.matmul(combined.I_inv_mass_derivatives[i], R.T)))
            external_force_impulse_mass_derivatives.append(np.array([0.,0.,0.]))
        #print("external force", external_force_impulse)
        shape_ground_apply_impulse_derivatives(combined, external_force_impulse, r, I_inv, None, external_force_impulse_mass_derivatives, I_inv_mass_derivatives)

    return external_force_contact_location
