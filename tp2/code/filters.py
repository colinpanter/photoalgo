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
    
    @staticmethod
    def filter(image: array, sigma: float) -> array:
        filter = GaussianFilter(sigma)
        return filter(image)
    
    def __call__(self, image: array) -> array:
        if len(image.shape) == 2:
            return gaussian_filter(image, self.sigma)
        else:
            channels = []
            for i in range(image.shape[2]):
                channels.append(gaussian_filter(image[:, :, i], self.sigma))
            return np.stack(channels, axis=2)
        return gaussian_filter(image, self.sigma)


class AntiGaussianFilter(Filter):
    def __init__(self, sigma:float) -> None:
        super().__init__()
        self.sigma = sigma
    
    def __call__(self, image: array) -> array:
        return image - gaussian_filter(image, self.sigma)


class GaussianSharpeningFilter(Filter):
    def __init__(self, sigma:float) -> None:
        super().__init__()
        self.sigma = sigma
    
    def __call__(self, image: array) -> array:
        return 2 * image - gaussian_filter(image, self.sigma)

class HighPassFilter(Filter):
    def __init__(self, cutoff:int) -> None:
        super().__init__()
        self.cutoff = cutoff
    
    def __call__(self, image: array) -> array:
        freq_img = np.fft.fftshift(np.fft.fft2(image))

        x, y = np.meshgrid(np.arange(freq_img.shape[1]), np.arange(freq_img.shape[0]))
        x = x - x.max() // 2
        y = y - y.max() // 2
        indexes = (x*x + y*y < self.cutoff*self.cutoff)

        freq_img[indexes] = 0
        return np.abs(ifft2(fftshift(freq_img)))

class LowPassFilter(Filter):
    def __init__(self, cutoff:int) -> None:
        super().__init__()
        self.cutoff = cutoff
    
    def __call__(self, image: array) -> array:
        freq_img = np.fft.fftshift(np.fft.fft2(image))

        x, y = np.meshgrid(np.arange(freq_img.shape[1]), np.arange(freq_img.shape[0]))
        x = x - x.max() // 2
        y = y - y.max() // 2
        indexes = (x*x + y*y > self.cutoff*self.cutoff)
        
        freq_img[indexes] = 0
        return np.abs(ifft2(fftshift(freq_img)))
