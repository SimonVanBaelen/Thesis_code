import matplotlib.pyplot as plt
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

"""
File used to create some illustrations used in my intermediate defense.
"""

def plotTimeSeries(s):
    x = [x for x in range(0, len(s))]
    plt.plot(x, s, linewidth=3)
    plt.xlabel('t')
    plt.show()

def plot_DTW_of_TimeSeries(s1, s2):
    path = dtw.warping_path(s1, s2)
    dtwvis.plot_warping(s1, s2, path, filename="dtw_plot.png")
    plt.xlabel('t')
    plt.show()

def plot_DTW_of_TimeSeries_example():
    s1 = np.array([5,-2,2,3,4,3,2,5,-2])
    s2 = np.array([-2,2,3,4,3,2,5,-2,1])
    path = dtw.warping_path(s1, s2)
    dtwvis.plot_warping(s1, s2, path, filename="dtw_plot.png")
    plt.xlabel('t')
    plt.show()

def plot_TimeSeries_window_animation(s, w):
    x = [x for x in range(0, len(s))]
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.plot(x, s, color="blue")
        ax.plot(x[i:i+w], s[i:i+w], 'r', linewidth=3)

        # Plot dotted lines
        axes = plt.gca()
        x_temp1 = [i , i]
        y_temp1 = [axes.get_ylim()[0], s[i]]
        x_temp2 = [i+w , i+w]
        y_temp2 = [axes.get_ylim()[0], s[i+w]]
        ax.plot(x_temp1, y_temp1, 'r:')
        ax.plot(x_temp2, y_temp2, 'r:')

    ani = FuncAnimation(fig, animate, frames=len(s)-w, interval=50, repeat=True)
    # plt.show()
    f = r"animation3.gif"
    writervideo = PillowWriter(fps=30)
    ani.save(f, writer=writervideo)


def plot_TimeSeries_window_animation2(s, w):
    x = [x for x in range(0, len(s))]
    fig, ax = plt.subplots()
    def animate(i):
        ax.clear()
        ax.set_ylim([-2, 2.5])
        if i > 4 and i < len(s)-4:
            ax.plot(x[i-4:i+w+4], s[i-4:i+w+4], color="black")
            ax.plot(x[i:i+w], s[i:i+w], 'b', linewidth=3)
            ax.plot(x[i-1:i+1], s[i-1:i+1], linewidth=3, color="red")
            ax.plot(x[i+w-1:i+w+1], s[i+w-1:i+w+1], linewidth=3, color="green")
            # Plot dotted lines
            axes = plt.gca()
            x_temp1 = [i , i]
            y_temp1 = [-2, s[i]]
            x_temp2 = [i+w , i+w]
            y_temp2 = [-2, s[i+w]]
            ax.plot(x_temp1, y_temp1, 'r:')
            ax.plot(x_temp2, y_temp2, 'r:')

    ani = FuncAnimation(fig, animate, frames=range(3, len(s)-w), interval=200, repeat=True)
    plt.show()
    f = r"animation3.gif"
    writervideo = PillowWriter(fps=5)
    ani.save(f, writer=writervideo)

def plot_TimeSeries_incremental_animation(s):
    x = [x for x in range(0, len(s))]
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.plot(x, s, color="blue")
        ax.plot(x[0:i], s[0:i], 'r', linewidth=3)

        # Plot dotted lines
        axes = plt.gca()
        x_temp1 = [i , i]
        y_temp1 = [-2, s[i]]
        ax.plot(x_temp1, y_temp1, 'r:')

    ani = FuncAnimation(fig, animate, frames=len(s), interval=100, repeat=True)
    plt.show()
    f = r"animation4.gif"
    writervideo = PillowWriter(fps=30)
    ani.save(f, writer=writervideo)

def plot_Infinite_TimeSeries_animation(s):
    x = [x for x in range(0, len(s))]
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.plot(x[0:i], s[0:i], color="blue")

    ani = FuncAnimation(fig, animate, frames=len(s)-1, interval=200, repeat=True)
    # plt.show()
    f = r"animationInfinite.gif"
    writervideo = PillowWriter(fps=30)
    ani.save(f, writer=writervideo)

def plot_multiple_graphs_on_same_graph(plots, x):
    for s in plots:
        plt.plot(x, s, linewidth=3)
    plt.show()
