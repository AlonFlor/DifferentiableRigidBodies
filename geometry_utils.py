import numpy as np


def normalize(vec):
    return vec / np.linalg.norm(vec)



#rotation-related geometry functions (except for joints)
def to_quaternion(vec):
    return np.append(vec,0)

def from_quaternion(quat):
    return quat[:3] #what if scalar part is nonzero?

def quaternion_mult(quat1, quat2):
    v1 = quat1[:3]
    v2 = quat2[:3]
    s1 = quat1[3]
    s2 = quat2[3]
    v_res = s1*v2 + s2*v1 + np.cross(v1, v2)
    s_res = s1*s2 - np.dot(v1, v2)
    return np.append(v_res, s_res)

def quaternion_inverse(quat):
    return np.append(-1*quat[:3], quat[3])

def rotation_from_quaternion(quat, vec):
    return from_quaternion(quaternion_mult(quat, quaternion_mult(to_quaternion(vec), quaternion_inverse(quat))))

def rotation_to_quaternion(angle, axis):
    vec = np.sin(angle/2)*axis
    return np.append(vec, np.cos(angle/2))

def angular_velocity(quat, quat_deriv):
    return from_quaternion(2*quaternion_mult(quat_deriv, quaternion_inverse(quat)))

def orientation_derivative(orientation_quat, angular_velocity):
    return 0.5*quaternion_mult(to_quaternion(angular_velocity), orientation_quat)

def quaternion_to_rotation_matrix(quat):
    result = np.ndarray((3,3))
    result[0][0] = 1. - 2*(quat[1]*quat[1] + quat[2]*quat[2])
    result[0][1] = 2*(quat[0]*quat[1] - quat[2]*quat[3])
    result[0][2] = 2*(quat[0]*quat[2] + quat[1]*quat[3])
    result[1][0] = 2*(quat[0]*quat[1] + quat[2]*quat[3])
    result[1][1] = 1. - 2*(quat[0]*quat[0] + quat[2]*quat[2])
    result[1][2] = 2*(quat[1]*quat[2] - quat[0]*quat[3])
    result[2][0] = 2*(quat[0]*quat[2] - quat[1]*quat[3])
    result[2][1] = 2*(quat[1]*quat[2] + quat[0]*quat[3])
    result[2][2] = 1. - 2*(quat[0]*quat[0] + quat[1]*quat[1])
    return result

def rotation_matrix_to_quaternion(R):
    result = None
    tr = R[0][0] + R[1][1] + R[2][2]
    if(tr > 0):
        r = np.sqrt(1. + tr)
        scale = 0.5 / r
        return np.array([(R[2][1] - R[1][2])*scale, (R[0][2] - R[2][0])*scale, (R[1][0] - R[0][1])*scale, 0.5*r])
    if R[0][0] >= R[1][1] and R[0][0] >= R[2][2]:
        r = np.sqrt(1. + R[0][0] - R[1][1] - R[2][2])
        scale = 0.5 / r
        return np.array([0.5*r, (R[1][0] + R[0][1])*scale, (R[2][0] + R[0][2])*scale, (R[2][1] - R[1][2])*scale])
    if R[1][1] >= R[0][0] and R[1][1] >= R[2][2]:
        r = np.sqrt(1. - R[0][0] + R[1][1] - R[2][2])
        scale = 0.5 / r
        return np.array([(R[0][1] + R[1][0])*scale, 0.5*r, (R[2][1] + R[1][2])*scale, (R[0][2] - R[2][0])*scale])
    #if R[2][2] >= R[0][0] and R[2][2] >= R[1][1]:
    r = np.sqrt(1. - R[0][0] - R[1][1] + R[2][2])
    scale = 0.5 / r
    return np.array([(R[0][2] + R[2][0])*scale, (R[1][2] + R[2][1])*scale, 0.5*r, (R[1][0] - R[0][1])*scale])

def to_world_coords(shape, vertex):
    return rotation_from_quaternion(shape.orientation, vertex - shape.COM) + shape.COM + shape.location

def rotate_only_to_world_coords(shape, vector):
    return rotation_from_quaternion(shape.orientation, vector)

def to_local_coords(shape, world_vertex):
    return rotation_from_quaternion(quaternion_inverse(shape.orientation), world_vertex - shape.COM - shape.location) + shape.COM

def rotate_only_to_local_coords(shape, vector):
    return rotation_from_quaternion(quaternion_inverse(shape.orientation), vector)

def velocity_of_point(shape, world_vertex):
    return shape.velocity + np.cross(shape.angular_velocity, world_vertex - shape.location - shape.COM)



#bounding-box-related geometry functions
def get_extrema(vertices):
    min_x = vertices[0][0]
    max_x = vertices[0][0]
    min_y = vertices[0][1]
    max_y = vertices[0][1]
    min_z = vertices[0][2]
    max_z = vertices[0][2]
    for vertex in vertices:
        if vertex[0] < min_x:
            min_x = vertex[0]
        elif vertex[0] > max_x:
            max_x = vertex[0]
        if vertex[1] < min_y:
            min_y = vertex[1]
        elif vertex[1] > max_y:
            max_y = vertex[1]
        if vertex[2] < min_z:
            min_z = vertex[2]
        elif vertex[2] > max_z:
            max_z = vertex[2]
    return min_x,max_x,min_y,max_y,min_z,max_z

def is_point_in_bounds(point, bounds):
    min_x,max_x,min_y,max_y,min_z,max_z = bounds
    in_bounds_x = (point[0] >= min_x) and (point[0] <= max_x)
    in_bounds_y = (point[1] >= min_y) and (point[1] <= max_y)
    in_bounds_z = (point[2] >= min_z) and (point[2] <= max_z)
    if in_bounds_x and in_bounds_y and in_bounds_z:
        return True
    return False

def get_indices_of_points_in_bounds(points, bounds):
    indices = []
    for i in np.arange(len(points)):
        if is_point_in_bounds(points[i], bounds):
            indices.append(i)
    return indices

def get_cylinder_bounding_box_extremum(cylinder, world_dir_in_local_coords, world_dir_index):
    #modify world dir so that it does not point outside of local z endpoints
    world_dir_in_local_coords[2] = 0.
    if np.linalg.norm(world_dir_in_local_coords) != 0:
        world_dir_in_local_coords = normalize(world_dir_in_local_coords)

    #test candidates
    candidate_1 = to_world_coords(cylinder, cylinder.end1 + world_dir_in_local_coords*cylinder.radius)[world_dir_index]
    min_val = max_val = candidate_1
    candidate_2 = to_world_coords(cylinder, cylinder.end1 - world_dir_in_local_coords*cylinder.radius)[world_dir_index]
    if candidate_2 < min_val:
        min_val = candidate_2
    if candidate_2 > max_val:
        max_val = candidate_2
    candidate_3 = to_world_coords(cylinder, cylinder.end2 + world_dir_in_local_coords*cylinder.radius)[world_dir_index]
    if candidate_3 < min_val:
        min_val = candidate_3
    if candidate_3 > max_val:
        max_val = candidate_3
    candidate_4 = to_world_coords(cylinder, cylinder.end2 - world_dir_in_local_coords*cylinder.radius)[world_dir_index]
    if candidate_4 < min_val:
        min_val = candidate_4
    if candidate_4 > max_val:
        max_val = candidate_4

    return min_val, max_val

def get_cylinder_bounding_box_extrema(cylinder):
    world_x_dir_in_local_coords = rotate_only_to_local_coords(cylinder, np.array([1.,0.,0.]))
    world_y_dir_in_local_coords = rotate_only_to_local_coords(cylinder, np.array([0.,1.,0.]))
    world_z_dir_in_local_coords = rotate_only_to_local_coords(cylinder, np.array([0.,0.,1.]))

    min_x, max_x = get_cylinder_bounding_box_extremum(cylinder, world_x_dir_in_local_coords, 0)
    min_y, max_y = get_cylinder_bounding_box_extremum(cylinder, world_y_dir_in_local_coords, 1)
    min_z, max_z = get_cylinder_bounding_box_extremum(cylinder, world_z_dir_in_local_coords, 2)

    #print(min_x,max_x,min_y,max_y,min_z,max_z)

    return min_x,max_x,min_y,max_y,min_z,max_z



#geometry functions for handling joints
def transformation_matrix(R, pos):
    return np.block([
        [R, pos.reshape(3,1)],
        [np.array([[0., 0., 0., 1.]])]
    ])

def to_transformation_matrix(quat, pos):
    R = quaternion_to_rotation_matrix(quat)
    return transformation_matrix(R, pos)

def from_transformation_matrix(M):
    R = M[:3, :3]
    p = M[:3, 3]
    quat = rotation_matrix_to_quaternion(R)
    return quat, p

def invert_transformation_matrix(M):
    R = M[:3,:3]
    p = M[:3, 3]
    return np.block([
        [R.T, -np.matmul(R.T, p.reshape(3,1))],
        [M[3, :]]
    ])

def cross_product_matrix_form(vec):
    return np.array([[0., -vec[2], vec[1]],
                     [vec[2], 0., -vec[0]],
                     [-vec[1], vec[0], 0.]])

def to_adjoint_matrix(M):
    R = M[:3, :3]
    p = M[:3, 3]
    return np.block([
        [R, np.zeros((3,3))],
        [np.matmul(cross_product_matrix_form(p), R), R]
    ])

def matrix_exponential_3x3(matrix, matrix_sq, theta):
    return np.identity(3) + matrix*np.sin(theta) + matrix_sq*(1. - np.cos(theta))

def matrix_exponential(screw_axis, theta):
    s_w = screw_axis[:3]
    s_v = screw_axis[3:]
    #case where s_w = 0, ||s_v|| = 1
    if np.linalg.norm(s_w) == 0:
        return np.block([
            [np.identity(3), np.dot(screw_axis,theta).reshape((3,1))],
            [0., 0., 0., 1.]
        ])
    #case where ||s_w|| = 1
    cp_sw = cross_product_matrix_form(s_w)
    cp_sw_sq = np.matmul(cp_sw, cp_sw)
    top_left = matrix_exponential_3x3(cp_sw, cp_sw_sq, theta)
    top_right_matrix = np.identity(3)*theta + (1. - np.cos(theta))*cp_sw + (theta - np.sin(theta))*cp_sw_sq
    top_right = np.matmul(top_right_matrix, s_v.reshape((3,1)))
    return np.block([
        [top_left, top_right],
        [0., 0., 0., 1.]
    ])

def hinge_joint_rotation_matrix(theta, hinge_location):
    c = np.cos(theta)
    s = np.sin(theta)
    E = np.array([[c, s, 0.],
                  [-s, c, 0.],
                  [0., 0., 1.]])
    zeros_matrix = np.zeros((3,3))
    X_J = np.block([
        [E, zeros_matrix],
        [zeros_matrix, E]
    ])
    X_T = np.block()

#joint subspace matrix definitions
hinge_joint_motion_subspace = np.array([0., 0., 1., 0., 0., 0.]).T
prismatic_joint_motion_subspace = np.array([0., 0., 0., 0., 0., 1.]).T
free_joint_motion_subspace = np.identity(6)



#functions for geometry involving points and lines
def line_segment_closest_point_from_other_point(other_point, end1, end2):
    v_e1 = other_point - end1
    v_e1_len = np.linalg.norm(v_e1)
    v_e2 = other_point - end2
    v_e2_len = np.linalg.norm(v_e2)
    line_segment = end2 - end1
    line_segment_len = np.linalg.norm(line_segment)
    line_segment_norm = normalize(line_segment)
    if np.dot(v_e1, line_segment_norm)*np.dot(v_e2, line_segment_norm) > 0:
        #other_point is off of line segment, check which endpoint is closest
        if v_e2_len > v_e1_len:
            return end1
        else:
            return end2
    #other_point is not off of line segment
    dist = v_e1_len*np.dot(normalize(v_e1), line_segment_norm)
    return end1 + dist*line_segment_norm

def point_on_line_segment_closest_to_line_segment_using_projection(LS1_end1, LS1_end2, LS2_end1, LS2_end2):
    #Gets point on LS2 closest to LS1
    #set LS1 to be the dot at the origin from the plane's point of view
    plane_normal = normalize(LS1_end2 - LS1_end1)

    #find the plane's vectors
    plane1 = np.cross(np.array([0.,0.,plane_normal[2]]), plane_normal)
    if(np.linalg.norm(plane1) == 0):
        plane1 = np.cross(np.array([0., plane_normal[1], 0.]), plane_normal)
    if(np.linalg.norm(plane1) == 0):
        plane1 = np.cross(np.array([plane_normal[0], 0., 0.]), plane_normal)
    if(np.linalg.norm(plane1) == 0):
        plane1 = np.cross(np.array([1., 1, 1.]), plane_normal)
    plane2 = np.cross(plane1, plane_normal)
    plane1 = normalize(plane1)
    plane2 = normalize(plane2)

    #project LS2's endpoints to the plane
    new_LS2_end1 = LS2_end1 - LS1_end1
    LS2_end1_plane1_coord = np.dot(new_LS2_end1, plane1)
    LS2_end1_plane2_coord = np.dot(new_LS2_end1, plane2)
    LS2_end1_plane_coords = np.array([LS2_end1_plane1_coord,LS2_end1_plane2_coord])
    new_LS2_end2 = LS2_end2 - LS1_end1
    LS2_end2_plane1_coord = np.dot(new_LS2_end2, plane1)
    LS2_end2_plane2_coord = np.dot(new_LS2_end2, plane2)
    LS2_end2_plane_coords = np.array([LS2_end2_plane1_coord,LS2_end2_plane2_coord])

    #take LS2's endpoints with respect to the plane normal for later
    LS2_end1_plane_normal = np.dot(new_LS2_end1, plane_normal)
    LS2_end2_plane_normal = np.dot(new_LS2_end2, plane_normal)
    plane_normal_coord = None

    #find the point on LS2's projection the plane closest to the origin (which is LS1)
    point_on_LS2_plane_coords = line_segment_closest_point_from_other_point(np.array([0.,0.]), LS2_end1_plane_coords, LS2_end2_plane_coords)
    
    LS2_on_plane = LS2_end2_plane_coords - LS2_end1_plane_coords
    LS2_on_plane_len = np.linalg.norm(LS2_on_plane)
    LS2_end1_plane_dist_from_origin = np.linalg.norm(LS2_end1_plane_coords)
    #assume LS1 and LS2 are not parallel and they do not intersect
    norm_LS2_on_plane = normalize(LS2_on_plane)
    dist = np.dot(-1*LS2_end1_plane_coords, norm_LS2_on_plane)
    if(dist < 0 or dist > LS2_on_plane_len):
        LS2_end1_dist = np.linalg.norm(LS2_end1_plane_coords)
        LS2_end2_dist = np.linalg.norm(LS2_end2_plane_coords)
        if(LS2_end1_dist < LS2_end2_dist):
            plane_normal_coord = LS2_end1_plane_normal
        else:
            plane_normal_coord = LS2_end2_plane_normal
    else:
        dist_from_LS2_end1_factor = dist/LS2_on_plane_len
        plane_normal_coord = (1-dist_from_LS2_end1_factor)*LS2_end1_plane_normal + dist_from_LS2_end1_factor*LS2_end2_plane_normal

    #project the result to world space
    result_point = point_on_LS2_plane_coords[0]*plane1 + point_on_LS2_plane_coords[1]*plane2 + plane_normal_coord*plane_normal + LS1_end1
    return result_point

def points_on_line_segments_closest_to_each_other(LS1_end1, LS1_end2, LS2_end1, LS2_end2):
    is_parallel_signifier = np.dot(normalize(LS1_end2 - LS1_end1), normalize(LS2_end2 - LS2_end1))
    if is_parallel_signifier==-1 or is_parallel_signifier==1:
        #lines are parallel, get midpoints. Assume line segments are inside the other if they were projected to a 1D axis parallel to both
        #answer is the midpoint of one line segment or the other, depending on which one is in inside the other
        axis = normalize(LS1_end2 - LS1_end1)

        #get vector from LS1 to LS2
        across_vector = LS2_end1 - LS1_end1
        LS1_to_LS2_vector = across_vector - np.dot(across_vector, axis)*axis
        
        LS1_end1_projected = np.dot(axis, LS1_end1)
        LS1_end2_projected = np.dot(axis, LS1_end2)
        LS2_end1_projected = np.dot(axis, LS2_end1)
        LS2_end2_projected = np.dot(axis, LS2_end2)

        LS1_min = None
        LS1_max = None
        if LS1_end1_projected > LS1_end2_projected:
            LS1_min = LS1_end2_projected
            LS1_max = LS1_end1_projected
        else:
            LS1_min = LS1_end1_projected
            LS1_max = LS1_end2_projected
        LS2_min = None
        LS2_max = None
        if LS2_end1_projected > LS2_end2_projected:
            LS2_min = LS2_end2_projected
            LS2_max = LS2_end1_projected
        else:
            LS2_min = LS2_end1_projected
            LS2_max = LS2_end2_projected

        if LS2_max < LS1_max:
            mid_LS2 = 0.5*(LS2_end1 + LS2_end2)
            LS1_equiv = mid_LS2 - LS1_to_LS2_vector
            return LS1_equiv, mid_LS2
        mid_LS1 = 0.5*(LS1_end1 + LS1_end2)
        LS2_equiv = mid_LS1 + LS1_to_LS2_vector
        return mid_LS1, LS2_equiv
    else:
        LS2_closest_point = point_on_line_segment_closest_to_line_segment_using_projection(LS1_end1, LS1_end2, LS2_end1, LS2_end2)
        LS1_closest_point = line_segment_closest_point_from_other_point(LS2_closest_point, LS1_end1, LS1_end2)
        return LS1_closest_point, LS2_closest_point



#functions to generate meshes for simplex convex shapes
def generate_box(dimensions):
    x_len, y_len, z_len = dimensions
    vertices = [np.array([-x_len/2, -y_len/2, -z_len/2]),
                np.array([x_len/2, -y_len/2, -z_len/2]),
                np.array([-x_len/2, y_len/2, -z_len/2]),
                np.array([x_len/2, y_len/2, -z_len/2]),
                np.array([-x_len/2, -y_len/2, z_len/2]),
                np.array([x_len/2, -y_len/2, z_len/2]),
                np.array([-x_len/2, y_len/2, z_len/2]),
                np.array([x_len/2, y_len/2, z_len/2])]
    face_normals = [np.array([0.,0.,-1]),
                    np.array([0.,0.,1]),
                    np.array([0.,-1.,0.]),
                    np.array([0.,1.,0.]),
                    np.array([-1.,0.,0.]),
                    np.array([1.,0.,0.])]
    faces = [([0, 1, 3, 2], 0),
             ([4, 5, 7, 6], 1),
             ([0, 1, 5, 4], 2),
             ([2, 3, 7, 6], 3),
             ([0, 2, 6, 4], 4),
             ([1, 3, 7, 5], 5)]
    return vertices, face_normals, faces

def generate_uncapped_cylinder(dimensions):
    radius, z_len = dimensions
    #vertices are separated by an angle of pi/16
    vertices = []
    face_normals = []
    for i in np.arange(32):
        theta = i*np.pi/16
        vertices.append(np.array([radius*np.cos(theta), radius*np.sin(theta), -1*z_len/2]))
        vertices.append(np.array([radius*np.cos(theta), radius*np.sin(theta), z_len/2]))

        theta_for_normal = theta + np.pi/32
        face_normals.append(normalize(np.array([radius*np.cos(theta_for_normal), radius*np.sin(theta_for_normal), 0.])))
    faces = []
    for i in np.arange(32):
        faces.append(([2*i, 2*i+1, (2*i+3)%64, (2*i+2)%64], i))
    return vertices, face_normals, faces

    
def generate_sphere(radius):
    vertices = []
    face_normals = [] #not generating these
    for i in np.arange(32):
        theta = i*np.pi/16
        for j in np.arange(1,32):
            phi = -1*j*np.pi/32
            vertices.append(radius*normalize(np.array([np.cos(theta)*np.sin(phi), np.cos(phi), np.sin(theta)*np.sin(phi)])))
    vertices.append(np.array([0., radius, 0.]))
    vertices.append(np.array([0., -1*radius, 0.]))
    len_vertices = len(vertices)
    
    faces = []
    for i in np.arange(32):
        faces.append(([len_vertices-2, i*31, ((i+1)%32)*31], None))
        for j in np.arange(30):
            faces.append(([i*31 + j, ((i+1)%32)*31 + j, ((i+1)%32)*31 + j+1, i*31 + j+1], None))
        faces.append(([len_vertices-1, i*31+30, ((i+1)%32)*31+30], None))
    return vertices, face_normals, faces
