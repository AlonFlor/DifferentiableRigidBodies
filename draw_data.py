import matplotlib.pyplot as plt

def plot_data(x_axis_data, y_axis_data):
    plt.plot(x_axis_data, y_axis_data, 'b')
    plt.show()

def plot_data(x_axis_data, y_axis_data1, y_axis_data2):
    plt.plot(x_axis_data, y_axis_data1, 'b', x_axis_data, y_axis_data2, 'g')
    plt.show()

def plot_data(x_axis_data, y_axis_data1, y_axis_data2, y_axis_data3):
    plt.plot(x_axis_data, y_axis_data1, 'b', x_axis_data, y_axis_data2, 'g', x_axis_data, y_axis_data3, 'r')
    plt.show()

def plot_3D_data(X_axis_data, y_axis_data, z_1, z_2, z_3, z_4):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X_axis_data, y_axis_data, z_1, color='b')
    ax.plot_wireframe(X_axis_data, y_axis_data, z_2, color='g')
    ax.plot_wireframe(X_axis_data, y_axis_data, z_3, color='r')
    ax.plot_wireframe(X_axis_data, y_axis_data, z_4, color='k')
    plt.show()

'''def plot_3D_data(X_axis_data, y_axis_data, z_1):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X_axis_data, y_axis_data, z_1, color='b')
    plt.show()'''

#plot_data([1,2,3,4], [1,4,9,16], [1,2,6,24])

#no overloading? These definitions are overriding each other? I do not like that.
