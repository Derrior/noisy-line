from attrs import define
import numpy as np

@define
class Cell:
    up: int
    down: int
    left: int
    right: int
    elements: np.ndarray
    def center(self):
        return (self.left + self.right) // 2, (self.up + self.down) // 2
