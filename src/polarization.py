import numpy as np
import matplotlib.pyplot as plt
from attrs import define
from abc import ABC, abstractmethod
from math import sqrt, atan2, hypot, sin, cos, pi
from typing import Optional, Tuple, List
from line_utils import abc2natural
from cell import Cell

class PolarizationMethod(ABC):
    @abstractmethod
    def __call__(self, cell: Cell):
        pass


class InertionPolarization(PolarizationMethod):
    def find_points_cloud_inertion(self, points: List[Tuple[int]]) -> Optional[Tuple[int]]:
        points = np.array(points)
        median = np.mean(points, axis=0)
        cov = np.cov(points.T)
        eig_values, eig_vectors = np.linalg.eig(cov)
        if eig_values[0] < eig_values[1]:
            eig_values[:] = eig_values[::-1]
            eig_vectors[:] = eig_vectors[::-1]

        direction = eig_vectors[1]
        normal = np.array([-direction[0], direction[1]])
        c = -normal.dot(median)
        a, b = normal
        if eig_values[1] != 0 and eig_values[0] / eig_values[1] < self.inertion: # there is no direction in data 
            return None
        if 0:
            print(eig_vectors)
            print(eig_values)
            print(median)

            print(points)
            print(a, b, c)
            x, y = zip(*points)
            plt.scatter(y, x, marker='o')
            
            xlim = plt.gca().get_xlim()
            ylim = plt.gca().get_ylim()
            xlist = np.linspace(xlim[0], xlim[1], 100) # Create 1-D arrays for x,y dimensions
            ylist = np.linspace(ylim[0], ylim[1], 100)
            X, Y = np.meshgrid(xlist, ylist) # Create 2-D grid xlist,ylist values
            F = a * Y + b * X + c
            plt.contour(X, Y, F, [0], colors = 'black', linestyles = 'solid')
            plt.gca().invert_yaxis()
            r, phi = abc2natural((a, b, c))
            draw_r_phi((r, phi))
            plt.show()
        ret = abc2natural((a, b, c))
        return ret

    def __init__(self, inertion=10, noise_points=10):
        self.inertion = inertion
        self.noise_points = noise_points
                    
    def __call__(self, cell: Cell) -> Optional[Tuple[int]]:
        points = []
        for i, row in enumerate(cell.elements):
            for j, elem in enumerate(row):
                if elem != 0:
                    points.append((i + cell.up, j + cell.left))
        if len(points) < self.noise_points:
            return None
        return self.find_points_cloud_inertion(points)

class MaxComponentInertionPolarization(InertionPolarization):
    Shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    MaxComponents = 5
    def __init__(self, inertion=10, noise_points=10):
        super(MaxComponentInertionPolarization, self).__init__(inertion, noise_points)

    def find_components(self, cell: Cell) -> List[List[int]]:
        cell_colors = {}
        current_color = 0
        for i, row in enumerate(cell.elements):
            for j, elem in enumerate(row):
                if elem != 0 and cell_colors.get((i, j), -1) == -1:
                    cell_colors[(i, j)] = current_color
                    queue = [(i, j)]
                    begin = 0
                    while begin < len(queue):
                        x, y = queue[begin]
                        for sx, sy in self.Shifts:
                            if 0 <= x + sx < cell.down - cell.up and \
                                0 <= y + sy < cell.right - cell.left and \
                                cell.elements[x + sx][y + sy] != 0 and \
                                cell_colors.get((x + sx, y + sy), -1) != current_color:
                                    cell_colors[(x + sx, y + sy)] = current_color
                                    queue.append((x + sx, y + sy))
                        begin += 1
                    current_color += 1

        components = [[] for i in range(current_color)]
        for coords, color in cell_colors.items():
            components[color].append(coords)
        return components

    def __call__(self, cell: Cell) -> Optional[Tuple[int]]:
        components = self.find_components(cell)
        if len(components) == 0:
            return None
        x = np.array(list(map(len, components)))
        max_color = np.argmax(x)
        points = components[max_color]
        if len(points) < self.noise_points:
            return None
        for i in range(len(points)):
            points[i] = (points[i][0] + cell.up, points[i][1] + cell.left)
        return self.find_points_cloud_inertion(points)
