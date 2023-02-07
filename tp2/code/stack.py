import numpy as np
from numpy import ndarray as array

from filters import GaussianFilter


class Stacker:
    def __init__(self, img:array, n_filters=6) -> None:
        self.gaussian = [img.copy()]
        self.laplacian = []

        sigma = 2
        for _ in range(n_filters):
            self.gaussian.append(GaussianFilter.filter(img, sigma))
            self.laplacian.append(self.gaussian[-2] - self.gaussian[-1])
            sigma *= 2
        
        self.laplacian.append(self.gaussian[-1])

        self.gaussian = np.stack(self.gaussian)
        self.laplacian = np.stack(self.laplacian)

