from peak_detection import PeakDetectionMethod
from line_utils import dist, unzip_pairs
import matplotlib.pyplot as plt
from math import pi, e, factorial, sqrt
from scipy.stats import norm
import numpy as np

class Edge:
    def __init__(self, length, u, v):
        self.length = length
        self.u = u
        self.v = v

    def __le__(self, other):
        if type(other) is int or float:
            return self.length <= other
        return self.length <= other.length
    
    def __lt__(self, other):
        if type(other) is int or float:
            return self.length < other
        return self.length < other.length

    def __ge__(self, other):
        if type(other) is int or float:
            return self.length >= other
        return self.length >= other.length
    
    def __gt__(self, other):
        if type(other) is int or float:
            return self.length > other
        return self.length > other.length

    def __eq__(self, other):
        if type(other) is int or float:
            return self.length == other
        return self.length == other.length

class Component:
    def __init__(self, tree_length, points):
        self.tree_length = tree_length
        self.size = len(points)
        self.points = points
    
    def show(self):
        x, y = unzip_pairs(self.points)
        plt.scatter(x, y, marker='.')


class TreeClustering(PeakDetectionMethod):
    def __init__(self, alpha, cutting_dist=1, rescale=1):
        """ 
        alpha - probability that cluster is accidental
        cutting_dist - distance for separating clusters
        rescale - scale for x axis. x / rescale and y are scaled similarly
        """
        
        self.alpha = alpha
        self.rescale = rescale
        self.cutting_dist = cutting_dist
        self.factorials = {}
        self.k0 = 0
        self.gamma = 0

    def calc_pi(self, n):
        k0, gamma = self.k0, self.gamma
        if n == 1:
            return e ** (-k0)
        f = self.factorials.get(n - 1, factorial(n - 1))
        self.factorials[n - 1] = f
        return k0 * (k0 + gamma * k0 * (n - 1)) ** (n - 2)\
                / f\
                * e ** -(k0 + gamma * k0 * (n - 1))

    def calc_bounding_box(self):
        bound = self.cutting_dist * 5
        min_x, min_y = self.point_list[0]
        max_x, max_y = min_x, min_y
        for x, y in self.point_list:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        self.bbox_shape = max_x - min_x + self.rescale * bound, max_y - min_y + bound
        print("bounding box size: ", self.bbox_shape)
        
    def find_minimal_tree(self):
        point_list = self.point_list
        self.dists = [0] * len(point_list)
        for i, p in enumerate(point_list):
            self.dists[i] = Edge(dist(p, point_list[0], self.rescale), 0, i)
        self.dists[0] = Edge(-1, -1, -1)
        self.edge_list = []
        self.total_tree_length = 0
        for i in range(len(point_list) - 1):
            best_d, best_j = 1e9, 0
            for j, d in enumerate(self.dists):
                if -1 < d < best_d:
                    best_d = d
                    best_j = j

            self.edge_list.append(self.dists[best_j])
            self.total_tree_length += self.dists[best_j].length
            self.dists[best_j] = Edge(-1, -1, -1)
            for j, p in enumerate(point_list):
                if self.dists[j] != -1:
                    new_dist = Edge(dist(p, point_list[best_j], self.rescale), best_j, j)
                    self.dists[j] = min(self.dists[j], new_dist)


    def find_components(self):
        self.components = []
        colors = [0] * len(self.point_list)
        color = 1
        for i in range(len(self.point_list)):
            if colors[i] == 0:
                colors[i] = color
                queue = [i]
                j = 0
                edges_length = 0
                while j < len(queue):
                    u = queue[j]
                    for edge in self.filtered_graph[u]:
                        if colors[edge.v] == 0:
                            edges_length += edge.length
                            queue.append(edge.v)
                            colors[edge.v] = color
                    j += 1
                color += 1
                self.components.append(Component(edges_length, [self.point_list[i] for i in queue]))
        
                        
    def create_tree_graph(self):
        self.filtered_graph = [[] for i in range(len(self.point_list))]
        for edge in self.filtered_edges:
            length, u, v = edge.length, edge.u, edge.v
            self.filtered_graph[u].append(edge)
            self.filtered_graph[v].append(Edge(length, v, u))

    def create_criteria(self):
        alpha_big_cluster = 0.01 * self.alpha
        alpha_long_cluster = self.alpha - alpha_big_cluster
        self.max_size = 1
        total_prob = 0
        pi_values = [0]
        while self.max_size < 100 and total_prob < (1 - alpha_big_cluster):
            total_prob += self.calc_pi(self.max_size)
            pi_values.append(self.calc_pi(self.max_size))
            self.max_size += 1

        plt.plot(list(range(self.max_size)), pi_values)
        plt.title("Pi(n) chart")
        plt.show()
        self.old_max_size = self.max_size
        self.max_size = 100

        print("cluster size limit: ", self.max_size, "; probability for small clusters", total_prob) 
        bin_probability = alpha_long_cluster / (self.max_size - 1)
        self.length_quantiles = [0] * self.max_size
        for size in range(2, self.max_size):
            clusters_amount = int(self.calc_pi(size) * len(self.point_list) / size) + 1
            # quantile for minimum of normal distributed values
            print("size: ", size, "clusters_amount: ", clusters_amount)
            print("parameters for ppf: ", "prob = ", (1 - bin_probability) ** (1/clusters_amount), "quantile =", 1 - (1 - bin_probability) ** (1/clusters_amount), "; mean =", self.edge_e * (size - 1), "; variance =", self.edge_v * (size - 1))
            """
            bin_prob = P(min(len) < q) = 1 - P(min(len) > q)
            1 - bin_prob = P(len > q) ** clusters_amount
            P(len > q) = (1 - bin_prob) ** (1 / clusters_amount)
            P(len < q) = 1 - (1 - bin_prob) ** (1 / clusters_amount)
            """
            quantile = norm.ppf(1 - (1 - bin_probability) ** (1/clusters_amount), loc=self.edge_e * (size - 1), scale=sqrt(self.edge_v * (size - 1)))
            print("quantile:", quantile)
            self.length_quantiles[size] = quantile
        
    def find_distribution_parameters(self):
        self.calc_bounding_box()
        self.density = len(self.point_list) / (self.bbox_shape[0] * self.bbox_shape[1])
        self.k0 = self.density * pi * self.cutting_dist ** 2 * self.rescale
        print("k0 =", self.k0)
        #self.k0 = 1.5
        self.gamma = 0.41 - 0.042 * self.k0
        
        edge_distribution = lambda x: (1 - e ** (-self.gamma * self.k0 * (x / self.cutting_dist)** 2)) / (1 - e ** (-self.gamma * self.k0))
        mean_agg = 0
        square_agg = 0
        distrib_v = []
        density_v = []
        last_prob = 0
        for i in np.linspace(0, self.cutting_dist, 1000):
            new_prob = edge_distribution(i)
            distrib_v.append(new_prob)
            density = (new_prob - last_prob)
            density_v.append(density)
            last_prob = new_prob
            mean_agg += i * density
            square_agg += i * i * density
        print(edge_distribution(self.cutting_dist))
        plt.plot( np.linspace(0, self.cutting_dist, 1000), distrib_v)
        plt.plot( np.linspace(0, self.cutting_dist, 1000), density_v)
        plt.show()
        square_agg -= mean_agg ** 2
        self.edge_e = mean_agg
        self.edge_v = square_agg
        return 
            

    def detect_peaks(self, point_list, max_amount):
        self.point_list = point_list
        self.find_distribution_parameters()
        self.find_minimal_tree()
        self.filtered_edges = list(filter(lambda e: e.length < self.cutting_dist, self.edge_list))
        self.create_tree_graph()
        self.find_components()
        self.create_criteria()
        self.components_to_return = list(filter(lambda c: c.size >= self.max_size or c.tree_length < self.length_quantiles[c.size], self.components))
        self.result = {}
        for c in self.components_to_return:
            x_center, y_center = 0, 0
            for p in c.points:
                x_center += p[0]
                y_center += p[1]
            x_center /= c.size
            y_center /= c.size
            self.result[(x_center, y_center)] = -c.size / (c.tree_length + 1) ** 6
        return self.result, {}

    def visualize(self):
        self.visualize_clusters()
        return
        for i in self.components:#_to_return:
            i.show()
    
    def visualize_clusters(self):
        size, length = [], []
        for c in self.components:
            size.append(c.size)
            length.append(c.tree_length)
        plt.scatter(size, length)

        ret_size, ret_length = [], []
        for c in self.components_to_return:
            ret_size.append(c.size)
            ret_length.append(c.tree_length)
        plt.scatter(ret_size, ret_length)
        plt.xlabel("size of cluster")
        plt.ylabel("total edges' length")
        #plt.show()

        xlim = list(plt.gca().get_xlim())
        ylim = list(plt.gca().get_ylim())
        plt.plot([self.old_max_size, self.old_max_size], ylim, color='black', label="max size")
        print("axes limits for scatter plots", xlim, ylim)
        plt.plot(list(range(len(self.length_quantiles))), self.length_quantiles, label="criteria curve")
        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.show()
