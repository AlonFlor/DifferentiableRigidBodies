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

def write_simulation_files(shapes, file, dir_name, dt, fps):
    outdata = open(file, "r", encoding="utf-8")
    data = outdata.read()
    outdata.close()
    
    lines = data.split("\n")
    frame_count = 1
    line_count = 0
    skip = round(1. / (dt * fps))
    for line in lines[1:-1]:
        if(line_count % skip != 0):
            line_count += 1
            continue
        data_row = line.split(",")
        for i in np.arange(len(data_row)):
            data_row[i] = float(data_row[i])
        out_file = open(os.path.join(dir_name,"frame_"+"{:0>4d}".format(frame_count)+".obj"), "w")
        time = data_row[0]
        out_file.write("#"+str(time)+"\n")
        shape_count = 0
        vertices_added = 0
        vertices_index_shift = [0]
        for shape in shapes:
            shape_data = data_row[1+7*shape_count:1+7*(shape_count+1)]
            loc = np.array(shape_data[:3])
            orientation = np.array(shape_data[3:])
            vertices_added = 0
            for vertex in shape.vertices:
                world_vertex = geometry_utils.rotation_from_quaternion(orientation, vertex - shape.COM) + shape.COM + loc #same as geometry_utils.to_world_coords but shape is not available since its past data is only in the file
                out_file.write("v")
                for coord in world_vertex:
                    out_file.write(" " + str(coord))
                out_file.write("\n")
                vertices_added += 1
            vertices_index_shift.append(vertices_added)
            shape_count += 1
        for i in np.arange(1,len(vertices_index_shift)):
            vertices_index_shift[i] += vertices_index_shift[i-1]
        shape_count = 0
        for shape in shapes:
            for vertex_indices,normal_index in shape.faces:
                out_file.write("f")
                for vertex_index in vertex_indices:
                    out_file.write(" "+str(vertex_index + vertices_index_shift[shape_count] + 1))
                out_file.write("\n")
            shape_count += 1
        out_file.close()
        frame_count += 1
        line_count += 1
