from math import pi, cos, sin, atan2, hypot
import numpy as np
import matplotlib.pyplot as plt
from cell import Cell

def abc2natural(line):
    a, b, c = line
    r = -c / hypot(a, b)
    phi = atan2(b, a)
    if r < 0:
        r = -r
        phi = phi + pi
    if phi < 0:
        phi += 2 * pi
    return (r, phi)

def natural2abc(line):
    r, phi = line
    a, b = cos(line[1]), sin(line[1])
    c = -line[0]
    return (a, b, c)


def draw_abc(line, intensity=1, cell=None, kwargs={}):
    default_parameters = {
        "colors": "red",
        "linestyles": "dotted"
    }
    default_parameters.update(kwargs)
    a, b, c = line
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    if cell == None:
        xlist = np.linspace(xlim[0], xlim[1], 1000) # Create 1-D arrays for x,y dimensions
        ylist = np.linspace(ylim[0], ylim[1], 1000)
    else:
        xlist = np.linspace(cell.left + 1, cell.right - 1, 2 * (cell.right - cell.left))
        ylist = np.linspace(cell.up + 1, cell.down - 1, 2 * (cell.down - cell.up))
    X, Y = np.meshgrid(xlist, ylist) # Create 2-D grid xlist,ylist values
    # swap axis for matplotlib
    F = a * Y + b * X + c
    plt.contour(X, Y, F, [0], alpha=max(1, intensity), **default_parameters)

def draw_r_phi(line, intensity=1, cell=None, kwargs={}):
    a, b = cos(line[1]), sin(line[1])
    c = -line[0]
    draw_abc((a, b, c), intensity, cell, kwargs)
