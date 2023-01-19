import numpy as np
from numpy import ndarray as array
from skimage import filters


class Metric:
    def calculate_score(self, img1:array, img2:array) -> float:
        raise NotImplementedError


class MSE(Metric):
    def calculate_score(self, img1:array, img2:array) -> float:
        diff = img1 - img2
        return (diff * diff).sum() / diff.size


class EdgeMSE(Metric):
    def calculate_score(self, img1:array, img2:array) -> float:
        diff = filters.sobel(img1) - filters.sobel(img2)
        return (diff * diff).sum() / diff.size