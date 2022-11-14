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

def plot_3D_data(x_axis_data, y_axis_data, z_1, z_2, z_3, z_4):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(x_axis_data, y_axis_data, z_1, color='b')
    ax.plot_wireframe(x_axis_data, y_axis_data, z_2, color='g')
    ax.plot_wireframe(x_axis_data, y_axis_data, z_3, color='r')
    ax.plot_wireframe(x_axis_data, y_axis_data, z_4, color='k')
    plt.show()

def plot_3D_loss_func(X_axis_data, y_axis_data, z_1):
    plt.contourf(X_axis_data, y_axis_data, z_1, 100, cmap='autumn')
    plt.colorbar()
    plt.show()

#plot_data([1,2,3,4], [1,4,9,16], [1,2,6,24])

#no overloading? These definitions are overriding each other? I do not like that.

'''size = 49
import numpy as np
loss_sweep_data = open("loss_sweep_file.csv", "r")
data = loss_sweep_data.read()
loss_sweep_data.close()
lines = data.split("\n")
X = np.zeros((size,size))
Y = np.zeros((size,size))
Z = np.zeros((size,size))
i = j = 0
for line in lines[1:-1]:
    x_next,y_next,z_next = line.strip().split(",")
    X[i,j] = x_next
    Y[i,j] = y_next
    Z[i,j] = z_next
    j += 1
    if j==size:
        i+=1
        j = 0

X = X.T
Y = Y.T
Z = Z.T

#X, Y = np.meshgrid(np.linspace(1., 11.5, 197), np.linspace(.5, 4., 197))

plot_3D_loss_func(X, Y, Z)'''