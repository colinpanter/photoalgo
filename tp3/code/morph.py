import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline

def morph(img1, img2, img1_pts, img2_pts, tri, warp_frac, dissolve_frac):
    morphed_pts = warp_frac * img1_pts + (1 - warp_frac) * img2_pts

    h, w = img1.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    pos = np.stack([x.flatten(), y.flatten()], axis=1)

    correspondance = np.zeros((h, w), dtype=int)

    tf1, tf2 = [], []
    for i, t in enumerate(tri):
        tri_morphed_pts = morphed_pts[t]
        tri_img1_pts = img1_pts[t]
        tri_img2_pts = img2_pts[t]

        inverse_morphed_matrix = np.linalg.inv(np.vstack([tri_morphed_pts.T, [1, 1, 1]]))
        tf1.append((np.vstack([tri_img1_pts.T, [1, 1, 1]]) @ inverse_morphed_matrix)[:2, :])
        tf2.append((np.vstack([tri_img2_pts.T, [1, 1, 1]]) @ inverse_morphed_matrix)[:2, :])

        shape = Path(tri_morphed_pts)
        inside = shape.contains_points(pos, radius=0.1).reshape((h, w))
        correspondance[inside] = i
    tf1, tf2 = np.array(tf1)[correspondance.flatten()], np.array(tf2)[correspondance.flatten()]
    
    morphed_pos1 = np.zeros(pos.shape) # (pos * tf1[:, :, 0] + pos[:, ::-1] * tf1[:, :, 1] + tf1[:, :, 2])
    morphed_pos1[:, 1] += pos[:, 0] * tf1[:, 0, 0] + pos[:, 1] * tf1[:, 0, 1] + tf1[:, 0, 2]
    morphed_pos1[:, 0] += pos[:, 0] * tf1[:, 1, 0] + pos[:, 1] * tf1[:, 1, 1] + tf1[:, 1, 2]
    morphed_pos2 = np.zeros(pos.shape) # (pos * tf2[:, :, 0] + pos[:, ::-1] * tf2[:, :, 1] + tf2[:, :, 2])
    morphed_pos2[:, 1] += pos[:, 0] * tf2[:, 0, 0] + pos[:, 1] * tf2[:, 0, 1] + tf2[:, 0, 2]
    morphed_pos2[:, 0] += pos[:, 0] * tf2[:, 1, 0] + pos[:, 1] * tf2[:, 1, 1] + tf2[:, 1, 2]
    
    morphed_img1 = np.dstack([(RectBivariateSpline(np.arange(h), np.arange(w), img1[:, :, i]).ev(morphed_pos1[:, 0], morphed_pos1[:, 1])).reshape((h, w)) for i in range(3)])
    morphed_img2 = np.dstack([(RectBivariateSpline(np.arange(h), np.arange(w), img2[:, :, i]).ev(morphed_pos2[:, 0], morphed_pos2[:, 1])).reshape((h, w)) for i in range(3)])
    
    return dissolve_frac * morphed_img1 + (1-dissolve_frac) * morphed_img2
