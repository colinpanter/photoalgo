import numpy as np

from splicer import Splicer
from metrics import Metric


class Optimizer():
    def optimize(self, splicer:Splicer) -> tuple[tuple, tuple]:
            raise NotImplementedError


class SingleScaleOptimizer(Optimizer):
    def __init__(self, metric:Metric, search_range:int=16) -> None:
        super().__init__()
        
        self.metric = metric
        self.search_range = search_range

    def optimize(self, splicer:Splicer) -> tuple[tuple, tuple]:

        min_error_rb, min_error_gb = np.inf, np.inf
        best_r, best_g = None, None
        for i in range(-self.search_range, self.search_range+1):
            for j in range(-self.search_range, self.search_range+1):
                h_slice = slice(abs(i), -abs(i)) if i != 0 else slice(None)
                v_slice = slice(abs(j), -abs(j)) if j != 0 else slice(None)
                img = splicer.splice(red_shift=(i, j), green_shift=(i, j), crop=0.1)[h_slice, v_slice]

                error_rb = self.metric.calculate_score(img[:, :, 0], img[:, :, 2])
                error_gb = self.metric.calculate_score(img[:, :, 1], img[:, :, 2])
                
                if error_rb < min_error_rb:
                    min_error_rb = error_rb
                    best_r = i, j

                if error_gb < min_error_gb:
                    min_error_gb = error_gb
                    best_g = i, j
        
        return best_r, best_g


class MultiScaleOptimizer(Optimizer):
    def __init__(self, metric:Metric, start_scale=16, search_range:int=16) -> None:
        super().__init__()

        scales = [start_scale]
        while scales[-1] > 1:
            scales.append(scales[-1] // 2)

        self.metric = metric
        self.scales = scales
        self.search_range = search_range

    def optimize(self, splicer:Splicer) -> tuple[tuple, tuple]:
        search_range = self.search_range
        r0 = 0, 0
        g0 = 0, 0
        for scale in self.scales:
            r0, g0 = (2*r0[0], 2*r0[1]), (2*g0[0], 2*g0[1])

            min_error_rb, min_error_gb = np.inf, np.inf
            best_r, best_g = None, None
            for i in range(-search_range, search_range+1):
                for j in range(-search_range, search_range+1):
                    h_slice = slice(abs(r0[0]+i), -abs(r0[0]+i)) if r0[0]+i != 0 else slice(None)
                    v_slice = slice(abs(r0[1]+j), -abs(r0[1]+j)) if r0[1]+j != 0 else slice(None)
                    img = splicer.splice(red_shift=(r0[0]+i, r0[1]+j), green_shift=(r0[0]+i, r0[1]+j), scale=scale, crop=0.1)[h_slice, v_slice]
                    error_rb = self.metric.calculate_score(img[:, :, 0], img[:, :, 2])
                    
                    h_slice = slice(abs(g0[0]+i), -abs(g0[0]+i)) if g0[0]+i != 0 else slice(None)
                    v_slice = slice(abs(g0[1]+j), -abs(g0[1]+j)) if g0[1]+j != 0 else slice(None)
                    img = splicer.splice(red_shift=(g0[0]+i, g0[1]+j), green_shift=(g0[0]+i, g0[1]+j), scale=scale, crop=0.1)[h_slice, v_slice]
                    error_gb = self.metric.calculate_score(img[:, :, 1], img[:, :, 2])
                    
                    if error_rb < min_error_rb:
                        min_error_rb = error_rb
                        best_r = r0[0]+i, r0[1]+j

                    if error_gb < min_error_gb:
                        min_error_gb = error_gb
                        best_g = g0[0]+i, g0[1]+j

            r0, g0 = best_r, best_g
            search_range = max(2, scale)
        
        return best_r, best_g
