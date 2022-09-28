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

#plot_data([1,2,3,4], [1,4,9,16], [1,2,6,24])

#no overloading? These definitions are overriding each other? I do not like that.
