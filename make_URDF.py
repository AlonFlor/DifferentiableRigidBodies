import file_handling

object_name = "hammer"
obj = file_handling.read_combined_boxes_rigid_body_file(object_name+".txt")

print(obj)

indent = 0

urdf_str = "<robot name = \"" + object_name + "\">\n"

indent = 1
for count,coord_list in enumerate(obj):
    x,y,z = coord_list
    urdf_str += "\t" * indent + "<link name = \"box" + str(count) + "\">\n"
    indent += 1

    urdf_str += "\t" * indent + "<visual>\n"
    indent +=1
    urdf_str += "\t" * indent + "<origin xyz(\"" + str(x) + " " + str(y) + " " + str(z) + "\") rpy(\"0 0 0\") />\n"
    urdf_str += "\t" * indent + "<geometry>\n"
    indent +=1
    urdf_str += "\t" * indent + "<box 1 />\n"
    indent -=1
    urdf_str += "\t" * indent + "</geometry>\n"
    indent -=1
    urdf_str += "\t" * indent + "</visual>\n"

    indent -=1
    urdf_str += "\t" * indent + "</link>\n"

    if count > 0:
        urdf_str += "\t" * indent + "<joint name = \"joint" + str(count) + "\" type = \"fixed\">\n"
        indent += 1
        urdf_str += "\t" * indent + "<parent link = \"box0\">\n"
        urdf_str += "\t" * indent + "<child link = \"box" + str(count) + "\">\n"
        indent -=1
        urdf_str += "\t" * indent + "</joint>\n"

urdf_str += "</robot>"

print(urdf_str)

file_handling.write_urdf(object_name, urdf_str)