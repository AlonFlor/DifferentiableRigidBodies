import file_handling

def write_URDF(object_name, destination_folder):
    obj = file_handling.read_combined_boxes_extra_info_rigid_body_file("object models\\" + object_name + ".txt")

    indent = 0

    urdf_str = "<robot name = \"" + object_name + "\">\n"

    indent = 1
    for count,info_list in enumerate(obj):
        x,y,z,mass,friction = info_list
        urdf_str += "\t" * indent + "<link name = \"box" + str(count) + "\">\n"
        indent += 1

        urdf_str += "\t" * indent + "<visual>\n"
        indent +=1
        urdf_str += "\t" * indent + "<origin xyz=\"" + str(x) + " " + str(y) + " " + str(z) + "\"/>\n"
        urdf_str += "\t" * indent + "<geometry>\n"
        indent +=1
        urdf_str += "\t" * indent + "<box size = \"1 1 1\"/>\n"
        indent -=1
        urdf_str += "\t" * indent + "</geometry>\n"
        indent -=1
        urdf_str += "\t" * indent + "</visual>\n"

        urdf_str += "\t" * indent + "<collision>\n"
        indent +=1
        urdf_str += "\t" * indent + "<origin xyz=\"" + str(x) + " " + str(y) + " " + str(z) + "\"/>\n"
        urdf_str += "\t" * indent + "<geometry>\n"
        indent +=1
        urdf_str += "\t" * indent + "<box size = \"1 1 1\"/>\n"
        indent -=1
        urdf_str += "\t" * indent + "</geometry>\n"
        indent -=1
        urdf_str += "\t" * indent + "</collision>\n"

        urdf_str += "\t" * indent + "<contact>\n"
        indent += 1
        urdf_str += "\t" * indent + "<lateral_friction value = \""+str(friction)+"\"/>\n"
        indent -= 1
        urdf_str += "\t" * indent + "</contact>\n"


        urdf_str += "\t" * indent + "<inertial>\n"
        indent +=1
        urdf_str += "\t" * indent + "<mass value = \""+str(mass)+"\"/>\n"
        urdf_str += "\t" * indent + "<inertia ixx=\""+str(mass/6.)+"\" ixy=\"0.0\" ixz=\"0.0\" iyy=\""+str(mass/6.)+"\" iyz=\"0.0\" izz=\""+str(mass/6.)+"\"/>\n"
        urdf_str += "\t" * indent + "<origin xyz=\"" + str(x) + " " + str(y) + " " + str(z) + "\"/>\n"
        indent -=1
        urdf_str += "\t" * indent + "</inertial>\n"

        indent -=1
        urdf_str += "\t" * indent + "</link>\n"

        if count > 0:
            urdf_str += "\t" * indent + "<joint name = \"joint" + str(count) + "\" type = \"fixed\">\n"
            indent += 1
            urdf_str += "\t" * indent + "<parent link = \"box0\"/>\n"
            urdf_str += "\t" * indent + "<child link = \"box" + str(count) + "\"/>\n"
            indent -=1
            urdf_str += "\t" * indent + "</joint>\n"

    urdf_str += "</robot>"

    #print(urdf_str)

    file_handling.write_urdf(destination_folder, object_name, urdf_str)
