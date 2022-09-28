import numpy as np
import geometry_utils

def AABB_intersect(shape1, shape2):
    a = shape1.bounding_box
    b = shape2.bounding_box
    return ((a[0] <= b[1]) and (b[0] <= a[1])) and ((a[2] <= b[3]) and (b[2] <= a[3])) and ((a[4] <= b[5]) and (b[4] <= a[5]))


#signed distance functions
def generate_signed_distance_function_for_face(center, vertices, normals, face):
    vertex_indices, normal_index = face
    normal = normals[normal_index]
    surface_dist = np.dot(vertices[vertex_indices[0]] - center, normal)
    return lambda local_coord : (np.dot(local_coord, normal) - surface_dist, normal)

def generate_signed_distance_function_for_faces(faces_sdf):
    return lambda local_coord : faces_sdf[np.argmax(np.array([face_sdf(local_coord)[0] for face_sdf in faces_sdf]))](local_coord)

def generate_signed_distance_function_sphere(radius):
    return lambda local_coord : (np.sqrt(np.dot(local_coord, local_coord)) - radius, local_coord / np.sqrt(np.dot(local_coord, local_coord)))

def generate_signed_distance_function_cylinder(radius):
    return lambda local_coord : (np.sqrt(np.dot(local_coord[:2], local_coord[:2])) - radius, np.append(local_coord[:2] / np.sqrt(np.dot(local_coord[:2], local_coord[:2])), 0.))



def min_max_points_along_axis(points, axis):
    min_val = np.dot(axis, points[0])
    max_val = min_val
    for point in points[1:]:
        candidate = np.dot(axis, point)
        if candidate < min_val:
            min_val = candidate
        if candidate > max_val:
            max_val = candidate
    return min_val, max_val

def separating_axis_intersect(shape1_points, shape2_points, dirs):
    for axis in dirs:
        min_shape1, max_shape1 = min_max_points_along_axis(shape1_points, axis)
        min_shape2, max_shape2 = min_max_points_along_axis(shape2_points, axis)
        if (max_shape1 < min_shape2) or (max_shape2 < min_shape1):
            return False
    return True

def translation_vector(shape1_points, shape2_points, axis):
    #assume shapes intersect in axis
    min_shape1, max_shape1 = min_max_points_along_axis(shape1_points, axis)
    min_shape2, max_shape2 = min_max_points_along_axis(shape2_points, axis)
    if min_shape1 <= max_shape2 and min_shape2 <= max_shape1:
        candidate_1 = max_shape2 - min_shape1
        candidate_2 = max_shape1 - min_shape2
        return min(candidate_1, candidate_2)
    else:
        print("error: shapes already not intersecting along this axis")
        exit()




def find_edge_contact_box_cylinder(v1_2, v2_2, cylinder):
    edge_closest_point, cylinder_axis_closest_point = geometry_utils.points_on_line_segments_closest_to_each_other(v1_2, v2_2, cylinder.end1, cylinder.end2)
    if cylinder.signed_distance(edge_closest_point)[0] < 0:
        return edge_closest_point
    return None

def find_edge_contact_box_sphere(v1_2, v2_2, sphere):
    edge_closest_point = geometry_utils.line_segment_closest_point_from_other_point(np.array([0.,0.,0.]), v1_2, v2_2)
    if sphere.signed_distance(edge_closest_point)[0] < 0:
        return edge_closest_point
    return None

def find_edge_contacts_box_rounded(box, rounded, vertex_sdf, rounded_is_cylinder):
    edges_contact_points = []
    for edge in box.edges:
        v1_index, v2_index = edge
        #v1_w = vertex_sdf[v1_index][3]
        #v2_w = vertex_sdf[v2_index][3]
        v1_2 = vertex_sdf[v1_index][4]
        v2_2 = vertex_sdf[v2_index][4]
        #if a vertex intersects, all of its edges are ignored
        v1_sdf = vertex_sdf[v1_index][1]
        v2_sdf = vertex_sdf[v2_index][1]
        if v1_sdf <=0 or v2_sdf <= 0:
            continue
        candidate = None
        if rounded_is_cylinder:
            candidate = find_edge_contact_box_cylinder(v1_2, v2_2, rounded)
        else:
            candidate = find_edge_contact_box_sphere(v1_2, v2_2, rounded)
        if candidate is not None:
            edges_contact_points.append(geometry_utils.to_world_coords(rounded,candidate))
    return edges_contact_points

def find_edge_contact_box_box(point, normalized_edge_vector, signed_distance, signed_distance_normal, stride_max_size):
    desired_stride = (signed_distance / (np.dot(normalized_edge_vector, -1*signed_distance_normal)))
    if desired_stride < 0:
        return None
    desired_stride_sign = np.sign(desired_stride)
    stride = min(stride_max_size, desired_stride)
    return point + stride*normalized_edge_vector

def find_edge_contact_wrapper_code_box_box(new_point, sdf_new_point, sdf_normal_new_point, other_point, box2_signed_distance):
    edge_vector = other_point - new_point
    normalized_edge_vector = geometry_utils.normalize(edge_vector)
    stride_max_size = np.linalg.norm(edge_vector)
    count = 1 #two tries allowed
    while count >= 0:
        if (np.abs(np.dot(normalized_edge_vector, sdf_normal_new_point)) > 0.00001): #threshold
            new_point = find_edge_contact_box_box(new_point, normalized_edge_vector, sdf_new_point, sdf_normal_new_point, stride_max_size)
            if new_point is not None:
                sdf_new_point, sdf_normal_new_point = box2_signed_distance(new_point)
                edge_vector = other_point - new_point
                if np.linalg.norm(edge_vector) < 0.00001: #threshold
                    break
                normalized_edge_vector = geometry_utils.normalize(edge_vector)
                stride_max_size = np.linalg.norm(edge_vector)
                if sdf_new_point <= 0:
                    return new_point, sdf_normal_new_point
            else:
                break
        count -= 1
    return None

def find_edge_contact_point_box_box(shape1, shape2, vertex_sdf, new_point, sdf_new_point, sdf_normal_new_point, other_point, v1_w, v2_w, v1_2, v2_2, new_point_first_direction_w):
    edge_result = find_edge_contact_wrapper_code_box_box(new_point, sdf_new_point, sdf_normal_new_point, other_point, shape2.signed_distance)
    if edge_result is not None:
        #add point from the other direction only if it does not the same as the point from the first direction
        new_point, sdf_normal_new_point = edge_result
        can_write = True
        if (np.linalg.norm(new_point-v1_2)<=0.00001) or (np.linalg.norm(new_point-v2_2)<=0.00001): #threshold
            can_write=False
        new_point_w = geometry_utils.to_world_coords(shape2, new_point)
        if new_point_first_direction_w is not None:
            if np.linalg.norm(new_point_w - new_point_first_direction_w) < 0.00001: #threshold
                can_write=False
        if can_write:
            return (new_point_w, v1_w, v2_w)
    return None

def find_edges_contact_points_box_box(shape1, shape2, vertex_sdf):
    edges_contact_points = []
    for edge in shape1.edges:
        v1_index, v2_index = edge
        #v1 = vertex_sdf[v1_index][0]
        #v2 = vertex_sdf[v2_index][0]
        v1_w = vertex_sdf[v1_index][3]
        v2_w = vertex_sdf[v2_index][3]
        v1_2 = vertex_sdf[v1_index][4]
        v2_2 = vertex_sdf[v2_index][4]
        #find edge contact point
        new_point = v1_2
        sdf_new_point = vertex_sdf[v1_index][1]
        sdf_normal_new_point = vertex_sdf[v1_index][2]
        edge_contact_point = find_edge_contact_point_box_box(shape1, shape2, vertex_sdf, new_point, sdf_new_point, sdf_normal_new_point, v2_2, v1_w, v2_w, v1_2, v2_2, None)
        if edge_contact_point is not None:
            edges_contact_points.append(edge_contact_point)
        #try edge contact point from the other direction
        new_point = v2_2
        sdf_new_point = vertex_sdf[v2_index][1]
        sdf_normal_new_point = vertex_sdf[v2_index][2]
        edge_contact_point = find_edge_contact_point_box_box(shape1, shape2, vertex_sdf, new_point, sdf_new_point, sdf_normal_new_point, v1_2, v1_w, v2_w, v1_2, v2_2, edge_contact_point)
        if edge_contact_point is not None:
            edges_contact_points.append(edge_contact_point)
        
    return edges_contact_points


def find_vertex_sdf(shape1, shape2):
    vertex_sdf = {}
    for vertex_index in np.arange(len(shape1.vertices)):
        vertex1 = shape1.vertices[vertex_index]
        vertex_w = geometry_utils.to_world_coords(shape1, vertex1)
        vertex2 = geometry_utils.to_local_coords(shape2, vertex_w)
        signed_dist, normal = shape2.signed_distance(vertex2)
        vertex_sdf[vertex_index] = (vertex1, signed_dist, normal, vertex_w, vertex2)
    return vertex_sdf


def get_box_box_contact_point_and_normal(shape1, shape2):
    #the normal is coming out of shape2
    #get vertex contact points
    vertex_sdf = find_vertex_sdf(shape1, shape2)
    vertex_contact_points = []
    for vertex1, signed_dist, normal, vertex_w, vertex2 in vertex_sdf.values():
        if signed_dist <= 0:
            vertex_contact_points.append((vertex_w, geometry_utils.rotate_only_to_world_coords(shape2, normal)))
    #get vertex contact points from the other shape
    vertex_sdf_1 = find_vertex_sdf(shape2, shape1)
    for vertex1, signed_dist, normal, vertex_w, vertex2 in vertex_sdf_1.values():
        if signed_dist <= 0:
            vertex_contact_points.append((vertex_w, geometry_utils.rotate_only_to_world_coords(shape1, normal)))
    #get edge contact points
    edge_contact_points = find_edges_contact_points_box_box(shape1, shape2, vertex_sdf)
    #get edge contact points from the other shape
    edge_contact_points += find_edges_contact_points_box_box(shape2, shape1, vertex_sdf_1)
    print(len(vertex_contact_points),"vertex contact"+("" if len(vertex_contact_points)==1 else "s"))
    print(len(edge_contact_points),"edge contact"+("" if len(edge_contact_points)==1 else "s"))
    #return nothing if no contact is found
    if len(vertex_contact_points) + len(edge_contact_points) == 0:
        return []
    #put all contacts in a list
    contacts = []
    contacts += vertex_contact_points
    contacts += edge_contact_points

    #consolidate contacts
    contact_points = []
    for data in contacts:
        point = data[0]
        contact_points.append(point)
    min_x,max_x,min_y,max_y,min_z,max_z = geometry_utils.get_extrema(contact_points)
    center_contact = 0.5*np.array([min_x+max_x, min_y+max_y, min_z+max_z])
    local_normal = None
    if len(edge_contact_points)==4 and len(vertex_contact_points)==0:
        #if there is a single edge-edge contact, the normal is the cross product of the edges
        edge1 = edge_contact_points[0][1] - edge_contact_points[0][2]
        edge2 = edge_contact_points[2][1] - edge_contact_points[2][2]
        local_normal = geometry_utils.normalize(np.cross(edge1, edge2))
        shape2_center_w = geometry_utils.to_world_coords(shape2, shape2.center)
        if np.linalg.norm(center_contact + local_normal - shape2_center_w) < np.linalg.norm(center_contact - local_normal - shape2_center_w):
            local_normal *= -1
        local_normal = geometry_utils.normalize(local_normal)
    elif len(vertex_contact_points)==1:
        #if there is a single vertex in the contacts, that verrtex decides the normal
        local_normal = vertex_contact_points[0][1]
    else:
        center_contact_1 = geometry_utils.to_local_coords(shape1, center_contact)
        center_contact_2 = geometry_utils.to_local_coords(shape2, center_contact)
        normal_candidate_1 = -1*geometry_utils.rotate_only_to_world_coords(shape1, shape1.signed_distance(center_contact_1)[1])
        normal_candidate_2 = geometry_utils.rotate_only_to_world_coords(shape2, shape2.signed_distance(center_contact_2)[1])
        if np.dot(normal_candidate_1,normal_candidate_2) == 1:
            local_normal = normal_candidate_2
        else:
            #MTV method. Expecting that both shapes intersect along both axes.
            #Also expecting that there is a small translation vector for the real normal and a huge translation vector for the other normal.
            shape1_local_points = []
            for vertex in shape1.vertices:
                shape1_local_points.append(geometry_utils.to_world_coords(shape1, vertex))
            shape2_local_points = []
            for vertex in shape2.vertices:
                shape2_local_points.append(geometry_utils.to_world_coords(shape2, vertex))
            candidate1 = translation_vector(shape1_local_points, shape2_local_points, normal_candidate_1)
            candidate2 = translation_vector(shape1_local_points, shape2_local_points, normal_candidate_2)
            if candidate1 < candidate2:
                local_normal = normal_candidate_1
            else:
                local_normal = normal_candidate_2

    #return the contact and normal
    return center_contact, local_normal


def get_sphere_box_face_contact_point_and_normal(box, sphere):
    #normal is coming out of sphere
    #assume only a box's face intersects the sphere
    sphere_center_in_box_coords = geometry_utils.to_local_coords(box, sphere.location)
    dist, reverse_direction = box.signed_distance(sphere_center_in_box_coords)
    if dist > sphere.radius:
        return []
    sphere_contact_point = sphere_center_in_box_coords - sphere.radius*reverse_direction
    box_contact_point = sphere_center_in_box_coords - dist*reverse_direction
    
    if box.signed_distance(box_contact_point)[0] > 0:
        #not a face
        return []
    sphere_contact_point_w = geometry_utils.to_world_coords(box, sphere_contact_point)
    box_contact_point_w = geometry_utils.to_world_coords(box, box_contact_point)
    center_contact = 0.5*(sphere_contact_point_w + box_contact_point_w)
    local_normal = geometry_utils.normalize(sphere_contact_point_w - box_contact_point_w) #due to interpenetration, normal is calculated as coming out of the box

    #check if there is contact, return nothing if there is no contact
    if np.linalg.norm(center_contact - sphere.location) > sphere.radius:
        return []
    
    return center_contact, local_normal

def get_box_round_shape_contact_point_and_normal(box, rounded):
    #normal is coming out of rounded
    rounded_is_cylinder = rounded.shape_type == "uncapped cylinder"
    
    #get vertex contact points
    vertex_sdf = find_vertex_sdf(box, rounded)
    vertex_contact_points = []
    for vertex1, signed_dist, normal, vertex_w, vertex2 in vertex_sdf.values():
        if signed_dist <= 0:
            vertex_contact_points.append(vertex_w)
    #get edge contact points
    edge_contact_points = find_edge_contacts_box_rounded(box, rounded, vertex_sdf, rounded_is_cylinder)
    print(len(vertex_contact_points),"vertex contact"+("" if len(vertex_contact_points)==1 else "s"))
    print(len(edge_contact_points),"edge contact"+("" if len(edge_contact_points)==1 else "s"))
    
    contacts = []
    contacts += vertex_contact_points
    contacts += edge_contact_points

    if len(contacts) == 0:
        if not rounded_is_cylinder:
            #box does not have any vertices nor edges contacting the rounded shape, so the rounded shape must be a sphere and the box's face must be intersecting
            print("no collisions, getting box-sphere contacts")
            return get_sphere_box_face_contact_point_and_normal(box, rounded)
        return []
    
    min_x,max_x,min_y,max_y,min_z,max_z = geometry_utils.get_extrema(contacts)
    center_contact = 0.5*np.array([min_x+max_x, min_y+max_y, min_z+max_z])
    center_contact_in_rounded_coords = geometry_utils.to_local_coords(rounded, center_contact)
    normal_rounded_coords = rounded.signed_distance(center_contact_in_rounded_coords)[1]
    normal = geometry_utils.to_world_coords(rounded, normal_rounded_coords)
    return center_contact, normal

def get_sphere_sphere_contact_point_and_normal(shape1, shape2):
    #the normal is coming out of shape2
    loc_sphere1 = shape1.location
    loc_sphere2 = shape2.location
    direction = geometry_utils.normalize(loc_sphere2 - loc_sphere1)
    sphere1_contact = loc_sphere1 + direction*shape1.radius
    sphere2_contact = loc_sphere2 - direction*shape2.radius
    center_contact = 0.5*(sphere1_contact + sphere2_contact)
    normal = -1*direction
    #no need to check if there is a contact, since this was done instead of checking oriented bounding boxes
    return center_contact, normal

def get_cylinder_cylinder_contact_point_and_normal(shape1, shape2):
    #the normal is coming out of shape2
    
    shape1_end1_w = geometry_utils.to_world_coords(shape1, shape1.end1)
    shape1_end2_w = geometry_utils.to_world_coords(shape1, shape1.end2)
    shape2_end1_w = geometry_utils.to_world_coords(shape2, shape2.end1)
    shape2_end2_w = geometry_utils.to_world_coords(shape2, shape2.end2)
    
    shape1_point_w, shape2_point_w = geometry_utils.points_on_line_segments_closest_to_each_other(shape1_end1_w, shape1_end2_w, shape2_end1_w, shape2_end2_w)
    shape1_cylindrical_projection = geometry_utils.normalize(shape2_point_w -  shape1_point_w)
    shape2_cylindrical_projection = -1*shape1_cylindrical_projection
    shape1_contact_point_w = shape1_point_w + shape1_cylindrical_projection*shape1.radius
    shape2_contact_point_w = shape2_point_w + shape2_cylindrical_projection*shape2.radius
    center_contact = 0.5*(shape1_contact_point_w + shape2_contact_point_w)
    
    #check if there is contact, return nothing if there is no contact
    if np.linalg.norm(center_contact - shape1_point_w) > shape1.radius or np.linalg.norm(center_contact - shape2_point_w) > shape2.radius:
        return []
    
    return center_contact, shape2_cylindrical_projection

def get_cylinder_sphere_contact_point_and_normal(cylinder, sphere):
    cylinder_end1_w = geometry_utils.to_world_coords(cylinder, cylinder.end1)
    cylinder_end2_w = geometry_utils.to_world_coords(cylinder, cylinder.end2)

    cylinder_point_w = geometry_utils.line_segment_closest_point_from_other_point(sphere.location, cylinder_end1_w, cylinder_end2_w)
    sphere_to_cylinder_vector = geometry_utils.normalize(cylinder_point_w - sphere.location)
    sphere_contact_point_w = sphere.location + sphere_to_cylinder_vector*sphere.radius
    cylinder_contact_point_w = cylinder_point_w - sphere_to_cylinder_vector*cylinder.radius
    center_contact = 0.5*(sphere_contact_point_w + cylinder_contact_point_w)

    #check if there is contact, return nothing if there is no contact
    if np.linalg.norm(center_contact - sphere.location) > sphere.radius:
        return []

    return center_contact, sphere_to_cylinder_vector


def shape_shape_collision_detection(collision_check_list):
    contacts = []
    for shape1, shape2 in collision_check_list:

        #sphere-sphere distance check, or if one of the shapes is not a sphere, oriented bounding box check
        can_be_collision = False
        if shape1.shape_type == "sphere" and shape2.shape_type == "sphere":
            if np.linalg.norm(shape1.location - shape2.location) < shape1.radius + shape2.radius:
                can_be_collision = True
        else:
            x_dir = np.array([1.,0.,0.])
            y_dir = np.array([0.,1.,0.])
            z_dir = np.array([0.,0.,1.])
            shape1_dirs = [geometry_utils.rotate_only_to_world_coords(shape1, x_dir), geometry_utils.rotate_only_to_world_coords(shape1, y_dir), geometry_utils.rotate_only_to_world_coords(shape1, z_dir)]
            shape2_dirs = [geometry_utils.rotate_only_to_world_coords(shape2, x_dir), geometry_utils.rotate_only_to_world_coords(shape2, y_dir), geometry_utils.rotate_only_to_world_coords(shape2, z_dir)]
            edge_axes = []
            for axis1 in shape1_dirs:
                for axis2 in shape2_dirs:
                    edge_axes.append(np.cross(axis1,axis2))
            all_dirs = shape1_dirs + shape2_dirs + edge_axes
            if shape1.shape_type=="sphere" or shape2.shape_type=="sphere":
                all_dirs.append(geometry_utils.normalize(shape1.location - shape2.location))
            shape1_points = []
            shape2_points = []
            for vertex in shape1.vertices:
                shape1_points.append(geometry_utils.to_world_coords(shape1, vertex))
            for vertex in shape2.vertices:
                shape2_points.append(geometry_utils.to_world_coords(shape2, vertex))
            can_be_collision = separating_axis_intersect(shape1_points, shape2_points, all_dirs)
            
        if can_be_collision:
            print("possible contact detected")
            reverse_normal = False
            if shape1.shape_type=="sphere" and shape2.shape_type=="sphere":
                #sphere-sphere
                this_pair_contact = get_sphere_sphere_contact_point_and_normal(shape1, shape2)
            elif shape1.shape_type=="box" and shape2.shape_type=="box":
                #box-box
                this_pair_contact = get_box_box_contact_point_and_normal(shape1, shape2)
            elif shape1.shape_type=="box":
                #box-rounded
                this_pair_contact = get_box_round_shape_contact_point_and_normal(shape1, shape2)
            elif shape2.shape_type=="box":
                #rounded-box
                this_pair_contact = get_box_round_shape_contact_point_and_normal(shape2, shape1)
                if len(this_pair_contact)>0:
                    reverse_normal = True
            elif shape1.shape_type=="uncapped cylinder" and shape2.shape_type=="uncapped cylinder":
                #cylinder-cylinder
                this_pair_contact = get_cylinder_cylinder_contact_point_and_normal(shape1, shape2)
            elif shape1.shape_type=="uncapped cylinder":
                #cylinder-sphere
                this_pair_contact = get_cylinder_sphere_contact_point_and_normal(shape1, shape2)
            elif shape2.shape_type=="uncapped cylinder":
                #sphere-cylinder
                this_pair_contact = get_cylinder_sphere_contact_point_and_normal(shape2, shape1)
                if len(this_pair_contact)>0:
                    reverse_normal = True
            else:
                print("error in shape type given")
                exit()

            if len(this_pair_contact)>0:
                #check for too much penetration
                if shape2.signed_distance(geometry_utils.to_local_coords(shape2, this_pair_contact[0]))[0] < -0.01:
                    print("penetration too deep")
                    print("penetration:",shape2.signed_distance(geometry_utils.to_local_coords(shape2, this_pair_contact[0]))[0])
                    print("threshold for this warning:", -0.01)
                    exit()

                if reverse_normal:
                    print("reversing normal to keep consistency")
                    this_pair_contact = (this_pair_contact[0], -1*this_pair_contact[1])
                
                print("this_pair_contacts", this_pair_contact)
                contacts.append(((shape1, shape2), this_pair_contact))
                #print("is this really a contact")
                #local_s1 = geometry_utils.to_local_coords(shape1, contact_info[0])
                #local_s2 = geometry_utils.to_local_coords(shape2, contact_info[0])
                #print(shape1.sdf(local_s1))
                #print(shape2.sdf(local_s2))
    return contacts

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

def shape_shape_collision_impulse(shape1, shape2, contact, dv, restitution):
    #the normal is coming out of shape2
    world_vertex, normal = contact
    
    #p_1 = geometry_utils.to_local_coords(shape1, world_vertex)
    #p_2 = geometry_utils.to_local_coords(shape2, world_vertex)
    r_1 = world_vertex - shape1.COM - shape1.location
    r_2 = world_vertex - shape2.COM - shape2.location

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




def shape_ground_collision_detection(ground_collisions):
    contacts = []
    for shape in ground_collisions:
        contact_location = None
        if shape.shape_type == "box":
            contact_points = []
            for vertex in shape.vertices:
                vertex_w = geometry_utils.to_world_coords(shape, vertex)
                if vertex_w[1] < -0.00001:   #threshold
                    contact_points.append(vertex_w)
            min_x,max_x,min_y,max_y,min_z,max_z = geometry_utils.get_extrema(contact_points)
            contact_location = 0.5*np.array([min_x+max_x, min_y+max_y, min_z+max_z])
        elif shape.shape_type == "uncapped cylinder":
            contact_location = shape.location - np.array([0., shape.radius, 0.]) #assume cylinder is lying on its side whenever it contacts the ground
        elif shape.shape_type == "sphere":
            contact_location = shape.location - np.array([0., shape.radius, 0.])
        else:
            print("error: unknown shape found in ground collision code")
            exit()
        contacts.append((shape, (contact_location, np.array([0., 1., 0.]))))

        print("shape.location",shape.location)
        print("contact_location",contact_location)

        #check for too much penetration
        if contact_location[1] < -0.01:
            print("penetration too deep")
            print("penetration:", contact_location[1])
            print("threshold for this warning:", -0.01)
            exit()
    return contacts

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
    r = world_vertex - shape.COM - shape.location

    R = geometry_utils.quaternion_to_rotation_matrix(shape.orientation)
    I_inv = np.matmul(R,np.matmul(shape.I_inv,R.T))
    
    rotational_part = np.dot(normal, np.cross(np.matmul(I_inv, np.cross(r,normal)), r))
    
    impulse_magn = (1. + restitution)*dv
    impulse_denom = 1./shape.mass + rotational_part
    impulse_magn = impulse_magn / impulse_denom

    impulse = impulse_magn*normal

    return impulse, r, I_inv

def shape_ground_apply_impulse(shape, impulse, r, I_inv):
    shape.velocity -= impulse / shape.mass
    shape.angular_velocity -= np.matmul(I_inv,np.cross(r,impulse))

    print("shape.velocity -= ",impulse / shape.mass)
    print("shape.angular_velocity -= ",np.matmul(I_inv,np.cross(r,impulse)))

    return



def get_shape_shape_tangential_velocity(shape1, shape2, contact_location, normal):
    shape1_velocity = geometry_utils.velocity_of_point(shape1, contact_location)
    shape2_velocity = geometry_utils.velocity_of_point(shape2, contact_location)

    shape1_velocity_normal = normal*np.dot(shape1_velocity, normal)
    shape2_velocity_normal = normal*np.dot(shape2_velocity, normal)

    shape1_velocity_perpendicular = shape1_velocity - shape1_velocity_normal
    shape2_velocity_perpendicular = shape2_velocity - shape2_velocity_normal

    return shape1_velocity_perpendicular - shape2_velocity_perpendicular

def get_shape_ground_tangential_velocity(shape, contact_location, normal):
    shape_velocity = geometry_utils.velocity_of_point(shape, contact_location)

    shape_velocity_normal = normal*np.dot(shape_velocity, normal)

    shape_velocity_perpendicular = shape_velocity - shape_velocity_normal

    return shape_velocity_perpendicular
