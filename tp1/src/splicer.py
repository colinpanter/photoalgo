import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt

from metrics import Metric


class Splicer():
    def __init__(self, r:np.ndarray, g:np.ndarray, b:np.ndarray) -> None:
        self.scales = {1: (r, g, b)}
    
    def splice(self, red_shift:tuple[int, int]=(0,0), green_shift:tuple[int, int]=(0,0), scale:int=1) -> np.ndarray:
        if scale not in self.scales:
            self.add_scale(scale)
        r, g, b = self.scales[scale]

        max_y_shift = max(red_shift[0], green_shift[0], 0)
        max_x_shift = max(red_shift[1], green_shift[1], 0)

        min_y_shift = abs(min(red_shift[0], green_shift[0], 0))
        min_x_shift = abs(min(red_shift[1], green_shift[1], 0))

        y_space = max_y_shift + min_y_shift
        x_space = max_x_shift + min_x_shift

        height = max((x.shape[0] for x in [r, g, b])) + y_space
        width = max((x.shape[1] for x in [r, g, b])) + x_space

        img = np.zeros((height+1, width+1, 3), dtype=float)

        r_height = y_space - (red_shift[0] + min_y_shift)
        r_width = x_space - (red_shift[1] + min_x_shift)
        img[r_height:r_height-y_space-1, r_width:r_width-x_space-1, 0] = r

        g_height = y_space - (green_shift[0] + min_y_shift)
        g_width = x_space - (green_shift[1] + min_x_shift)
        img[g_height:g_height-y_space-1, g_width:g_width-x_space-1, 1] = g

        b_height = y_space - min_y_shift
        b_width = x_space - min_x_shift
        img[b_height:b_height-y_space-1, b_width:b_width-x_space-1, 2] = b

        return img

    def add_scale(self, scale:int) -> None:
        r, g, b = [downscale_local_mean(channel, scale) for channel in self.scales[1]]
        self.scales[scale] = (r, g, b)
