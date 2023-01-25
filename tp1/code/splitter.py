import numpy as np


class Splitter():
    def __init__(self) -> None:
        pass

    def split(self, img:np.ndarray, crop:int=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        h, w = img.shape
        h = h // 3
        b, g, r = img[:h], img[h:2*h], img[2*h:3*h]

        if isinstance(crop, int):
            r, g, b = r[crop:-crop, crop:-crop], g[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
        elif isinstance(crop, float):
            h_crop, w_crop = int(crop * h), int(crop * w)
            r, g, b = r[h_crop:-h_crop, w_crop:-w_crop], g[h_crop:-h_crop, w_crop:-w_crop], b[h_crop:-h_crop, w_crop:-w_crop]

        return r, g, b
