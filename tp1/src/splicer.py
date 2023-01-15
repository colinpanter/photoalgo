import numpy as np
import matplotlib.pyplot as plt


class Splicer():
    def __init__(self, r:np.ndarray, g:np.ndarray, b:np.ndarray) -> None:
        self.r = r
        self.g = g
        self.b = b
    
    def splice(self, red_shift:tuple=(0,0), green_shift:tuple=(0,0)) -> np.ndarray:
        max_y_shift = max(red_shift[0], green_shift[0], 0)
        max_x_shift = max(red_shift[1], green_shift[1], 0)

        min_y_shift = abs(min(red_shift[0], green_shift[0], 0))
        min_x_shift = abs(min(red_shift[1], green_shift[1], 0))

        y_space = max_y_shift + min_y_shift
        x_space = max_x_shift + min_x_shift

        height = max((x.shape[0] for x in [self.r, self.g, self.b])) + y_space
        width = max((x.shape[1] for x in [self.r, self.g, self.b])) + x_space

        img = np.zeros((height+1, width+1, 3), dtype=int)

        r_height = y_space - (red_shift[0] + min_y_shift)
        r_width = x_space - (red_shift[1] + min_x_shift)
        img[r_height:r_height-y_space-1, r_width:r_width-x_space-1, 0] = self.r

        g_height = y_space - (green_shift[0] + min_y_shift)
        g_width = x_space - (green_shift[1] + min_x_shift)
        img[g_height:g_height-y_space-1, g_width:g_width-x_space-1, 1] = self.g

        b_height = y_space - min_y_shift
        b_width = x_space - min_x_shift
        img[b_height:b_height-y_space-1, b_width:b_width-x_space-1, 2] = self.b

        return img

    def optimize(self):
        min_error_rb, min_error_gb = np.inf, np.inf
        best_r, best_g = None, None
        for i in range(-15, 16):
            for j in range(-15, 16):
                h_slice = slice(abs(i), -abs(i)) if i != 0 else slice(None)
                v_slice = slice(abs(j), -abs(j)) if j != 0 else slice(None)
                img = self.splice(red_shift=(i, j), green_shift=(i, j))[h_slice, v_slice]

                size = img.size / 3

                error_rb = np.square(img[:, :, 0] - img[:, :, 2]).sum() / size
                error_gb = np.square(img[:, :, 1] - img[:, :, 2]).sum() / size

                if error_rb < min_error_rb:
                    min_error_rb = error_rb
                    best_r = (i, j)

                if error_gb < min_error_gb:
                    min_error_gb = error_gb
                    best_g = (i, j)
        
        return best_r, best_g
