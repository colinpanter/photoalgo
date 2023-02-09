import numpy as np
from numpy import ndarray as array
from numpy.fft import fft2, ifft2, fftshift, ifftshift


class Stacker:
    def __init__(self, img:array, n_filters=6, start=0.5) -> None:
        self.gaussian = [img.copy()]
        self.laplacian = []

        sigmas = [start/(1 << 2*i) for i in range(n_filters-1)]

        fft_img = fftshift(fft2(img, axes=(0, 1)), axes=(0, 1))

        x, y = np.meshgrid(np.arange(fft_img.shape[1]), np.arange(fft_img.shape[0]))
        x_max, y_max = x.max() // 2, y.max() // 2
        x = (x - x_max) / x_max
        y = (y - y_max) / y_max
        in_exp = -(x * x + y * y) / 2
        if len(fft_img.shape) > 2:
            in_exp = np.stack([in_exp] * 3, axis=2)
        for s in sigmas:
            gaussian = np.exp(in_exp / (s*s))
            filtered_img = np.abs(ifft2(ifftshift(fft_img * gaussian, axes=(0, 1)), axes=(0, 1)))

            self.gaussian.append(filtered_img)
            self.laplacian.append(self.gaussian[-2] - self.gaussian[-1])
        
        self.laplacian.append(self.gaussian[-1])

        self.gaussian = np.stack(self.gaussian)
        self.laplacian = np.stack(self.laplacian)

