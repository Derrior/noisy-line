from math import sin, cos, pi, inf
import numpy as np
import cv2 as cv
from attrs import define, field


def get_min_max_bound(center, shift, a, b):
    """
    solve double inequality a < center + t * shift < b
    """
    if a > b:
        return 0, 0
    if shift == 0:
        if a < center < b:
            return (-inf, inf)
        return 0, 0
    t_0 = (a - center) / shift
    t_1 = (b - center) / shift
    if t_0 > t_1:
        return t_1, t_0
    return t_0, t_1

@define
class Line:
    r: float
    phi: float
    width: float = field(default=1)
    noise_level: float = field(default=0)

def generate_line(img, line, value=0):
    """
    modifies image: draws a line with given value
    """

    normal = np.array([cos(line.phi), sin(line.phi)])
    h, w = img.shape
    center = line.r * normal
    line_vec = np.array([sin(line.phi), -cos(line.phi)])
    # count bounding_box
    min_t_h, max_t_h = get_min_max_bound(center[0], line_vec[0], 0, h - 1)
    min_t_w, max_t_w = get_min_max_bound(center[1], line_vec[1], 0, w - 1)
    
    for t in np.linspace(max(min_t_h, min_t_w), min(max_t_h, max_t_w), h + w):
        for s in np.linspace((-line.width + 1) / 2, line.width / 2, line.width):
            noise = np.random.normal(scale=line.noise_level, size=2)
            x, y = center + line_vec * t + normal * s + noise
            x, y = int(x), int(y)
            if 0 <= x < h and 0 <= y < w:
                img[int(x)][int(y)] = value


def apply_general_noise(img, noise_sampler, ratio):
    h, w = img.shape
    for i in range(int(h * w * ratio)):
        x, y = noise_sampler()
        if 0 <= x < h and 0 <= y < w:
            img[int(x)][int(y)] = 0

        
def generate_image(lines, h=200, w=300):
    img = np.ones((h, w), dtype=np.uint8) * 255
    colors = [0, 50, 100]
    for i, line in enumerate(lines):
        generate_line(img, line, colors[i % len(colors)])
    return img

"""
cv.imshow("../test-images/1.png", generate_image([Line(1, 1, 1)]))
cv.imshow("../test-images/2.png", generate_image([Line(100, 1, 1, 1)]))
cv.imshow("../test-images/3.png", generate_image([Line(100, 1, 1), Line(95, 1, 1), Line(90, 1, 1)]))
cv.imshow("../test-images/4.png", generate_image([Line(100, 0, 5), Line(100, 0.01, 1), Line(100, -0.01, 1)]))
cv.waitKey(0)
"""

