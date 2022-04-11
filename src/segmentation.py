from polarization import PolarizationMethod
from cell import Cell
from math import pi

def grid_from_image(img, grid_size=60):
    grid = []
    for i in range(0, img.shape[0], grid_size):
        grid.append([])
        for j in range(0, img.shape[1], grid_size):
            grid[-1].append(Cell(
                i,
                min(i + grid_size, img.shape[0]),
                j,
                min(j + grid_size, img.shape[1]),
                img[i:i + grid_size,j: j + grid_size]))
    return grid

def segments_detection(grid, method: PolarizationMethod):
    segments = []
    for row in grid:
        for cell in row:
            result = method(cell)
            if result != None:
                segments.append((result, cell))
    return segments

def loop_segment_list(segments, shift=2 * pi):
    for i in range(len(segments)):
        elem = segments[i]
        if elem[1] < shift:
            segments.append((elem[0], elem[1] + 2 * pi))
    return segments

