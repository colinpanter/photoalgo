import numpy as np
from skimage import img_as_float
from skimage.transform import downscale_local_mean
from skimage.io import imread


class Splicer():
    def __init__(self, r:np.ndarray, g:np.ndarray, b:np.ndarray) -> None:
        self.scales = {1: (r, g, b)}
    
    def splice(self, red_shift:tuple[int, int]=(0,0), green_shift:tuple[int, int]=(0,0), scale:int=1, crop:float|int=None) -> np.ndarray:
        if scale not in self.scales:
            self.add_scale(scale)
        r, g, b = self.scales[scale]

        if isinstance(crop, int):
            r, g, b = r[crop:-crop, crop:-crop], g[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
        elif isinstance(crop, float):
            h_crop, w_crop = int(crop * r.shape[0]), int(crop * r.shape[1])
            r, g, b = r[h_crop:-h_crop, w_crop:-w_crop], g[h_crop:-h_crop, w_crop:-w_crop], b[h_crop:-h_crop, w_crop:-w_crop]

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
    
    def split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.scales[1]
    
    def get_original(self) -> np.ndarray:
        raise NotImplementedError

    def add_scale(self, scale:int) -> None:
        r, g, b = [downscale_local_mean(channel, scale) for channel in self.scales[1]]
        self.scales[scale] = (r, g, b)


class GorskiiSplicer(Splicer):
    def __init__(self, image_path:str) -> None:
        image = img_as_float(imread(image_path))
        self.original = image

        h, w = image.shape
        h = h // 3
        b, g, r = image[:h], image[h:2*h], image[2*h:3*h]

        super().__init__(r, g, b)
    
    def get_original(self) -> np.ndarray:
        return self.original


class ThreeRGBSplicer(Splicer):
    def __init__(self, r_path:str, g_path:str, b_path:str) -> None:
        r = img_as_float(imread(r_path))
        g = img_as_float(imread(g_path))
        b = img_as_float(imread(b_path))

        self.original = np.vstack([b, g, r])
        super().__init__(r[:, :, 0], g[:, :, 1], b[:, :, 2])
    
    def get_original(self) -> np.ndarray:
        return self.original
