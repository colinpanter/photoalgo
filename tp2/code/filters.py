import numpy as np
from numpy import ndarray as array
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter


class Filter:
    def __call__(self, image:array) -> array:
        raise NotImplementedError


class GaussianFilter(Filter):
    def __init__(self, sigma:float) -> None:
        super().__init__()
        self.sigma = sigma
    
    def __call__(self, image:array) -> array:
        if len(image.shape) == 2:
            return gaussian_filter(image, self.sigma)
        else:
            channels = []
            for i in range(image.shape[2]):
                channels.append(gaussian_filter(image[:, :, i], self.sigma))
            return np.stack(channels, axis=2)


class AntiGaussianFilter(Filter):
    def __init__(self, sigma:float) -> None:
        super().__init__()
        self.sigma = sigma
    
    def __call__(self, image:array) -> array:
        if len(image.shape) == 2:
            return image - gaussian_filter(image, self.sigma)
        else:
            channels = []
            for i in range(image.shape[2]):
                channels.append(gaussian_filter(image[:, :, i], self.sigma))
            return image - np.stack(channels, axis=2)
