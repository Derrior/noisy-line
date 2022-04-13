from attrs import define
import numpy as np
from math import pi, log
import matplotlib.pyplot as plt
import cv2 as cv

from polarization import PolarizationMethod, InertionPolarization
from peak_detection import PeakDetectionMethod, NausWallenstein
from segmentation import grid_from_image, segments_detection, loop_segment_list
from line_utils import draw_r_phi



def is_neighbour(one, other):
    return abs(one[0] - other[0]) < 40 and abs(one[1] - other[1]) < 0.02

@define
class ExperimentData:
    # original image
    img: np.ndarray
    # edges extracted from __img__
    edges: np.ndarray
    # segments extracted with grid from __edges__
    segments: list
    # peaks -- detected clusters of segments, likely one peak is one line
    peaks: dict
    # windows triggered peak detection algorithm
    windows: dict
    # deduplicated lines
    filtered_lines: list
        
    def __init__(self):
        self.segments = []
        self.peaks = {}
        self.windows = {}
        self.filtered_lines = []
    
    def draw_edges(self):
        plt.imshow(self.edges, cmap='gray')

    def draw_scatter_lines(self):
        r, phi = [], []
        for s, cell in self.segments:
            r.append(s[0])
            phi.append(s[1])
        plt.scatter(r, phi, alpha=0.3)
        
    def draw_segments(self):
        plt.imshow(self.edges, cmap='gray')
        for segment in self.segments:
            color = 'yellow'
            for l, score in self.filtered_lines:
                if is_neighbour(l, segment[0]):
                    color = 'red'
            draw_r_phi(segment[0], cell=segment[1], kwargs={"linestyles" : "solid", "colors" : color})
    def draw_windows_dist(self):
        x, y, z = [], [], []
        items = list(self.windows.items())
        items.sort()
        for k, v in items:
            if v[1] > 0:
                x.append(k)
                y.append(v[0])
                z.append(-log(v[1]))
        plt.xlabel("angular line parameter")
        plt.plot(x, y, label='points in window')
        plt.plot(x, z, label='statistic log value for window')
        plt.legend()

    def draw_lines(self):
        plt.imshow(self.img, cmap='gray', alpha=0.5)
        plt.xlim(-10, self.img.shape[1] + 10)
        plt.ylim(-10, self.img.shape[0] + 10)
        plt.gca().invert_yaxis()
        for i in range(len(self.filtered_lines)):
            line, score = self.filtered_lines[i]
            draw_r_phi(line, 1 - (i * 0.5 / len(self.filtered_lines)))
        
    def visualize(self, compact=False):
        if compact:
            plt.figure(figsize=(18, 10))
            plt.subplot(221)
            self.draw_scatter_lines()
            plt.subplot(222)
            self.draw_segments()
            plt.subplot(223)
            self.draw_lines()
            plt.subplot(224)
            self.draw_windows_dist()
            plt.show()
        else:
            plt.figure(figsize=(20, 20))
            self.draw_scatter_lines()
            plt.show()
            plt.figure(figsize=(20, 20))
            self.draw_segments()
            plt.show()
            plt.figure(figsize=(20, 20))
            self.draw_lines()
            plt.show()
            plt.figure(figsize=(20, 20))
            self.draw_windows_dist()
            plt.show()
    

def run_exp(img, grid_size=20,
            polarization_method: PolarizationMethod=InertionPolarization(),
            peak_detection_method: PeakDetectionMethod=NausWallenstein(),
            lines_count=10):
    
    exp_data = ExperimentData()
    exp_data.img = img
    edges = cv.Canny(img, 50, 400)
    exp_data.edges = edges
    
    grid = grid_from_image(edges, grid_size)
    segments = segments_detection(grid, polarization_method)
    exp_data.segments = segments
    
    if len(segments) == 0:
        print("no segments on image")
        return exp_data

    segment_list = list(map(lambda x: x[0], segments))
    segment_list = loop_segment_list(segment_list, pi * 0.1)

    peaks, windows = peak_detection_method.detect_peaks(segment_list, 5)
    exp_data.peaks, exp_data.windows = peaks, windows

    by_neighbour_amount = list(peaks.items())
    by_neighbour_amount.sort(key=lambda x: x[1])    

    drawn = set()
    exp_data.filtered_lines = []
    i = 0
    while i < len(by_neighbour_amount) and len(drawn) < lines_count:
        is_drawn = False
        for other in drawn:
            if is_neighbour(by_neighbour_amount[i][0], other):
                is_drawn = True
        if is_drawn:
            i += 1
            continue
        exp_data.filtered_lines.append(by_neighbour_amount[i])
        drawn.add(by_neighbour_amount[i][0])
        i += 1    
    return exp_data


