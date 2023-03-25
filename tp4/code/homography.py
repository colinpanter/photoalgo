import numpy as np
from numpy import ndarray as array
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from scipy.interpolate import RectBivariateSpline


def appliqueTransformation(img:array, H:array, lim:bool=False) -> array:
    h, w, _ = img.shape

    corners = np.array([[0, 0, 1], [0, h, 1], [w, 0, 1], [w, h, 1]]).T

    corners_tf:array = H @ corners
    corners_tf = corners_tf / corners_tf[2]
    corners_tf = corners_tf.astype(int)

    x_min, y_min, _ = corners_tf.min(axis=1)
    x_max, y_max, _ = corners_tf.max(axis=1)
    
    x, y = np.meshgrid(np.arange(x_max - x_min + 1), np.arange(y_max - y_min + 1))
    x, y = x + x_min, y + y_min
    
    pts = np.stack([x, y, np.ones(x.shape)])
    H_inv = np.linalg.inv(H)

    correspondance =  (H_inv @ pts.reshape((3, -1))).reshape(pts.shape)
    correspondance = correspondance / correspondance[2]

    img_tf = np.dstack([(RectBivariateSpline(np.arange(h), np.arange(w), img[:, :, i]).ev(correspondance[1], correspondance[0])).reshape(correspondance.shape[1:]) for i in range(3)])
    img_tf[np.logical_or(np.logical_or(correspondance[0] < 0, correspondance[0] > w), np.logical_or(correspondance[1] < 0, correspondance[1] > h))] = 0

    return (img_tf, (x_min, x_max+1, y_min, y_max+1)) if lim else img_tf


def calculate_homography(source_pts: array, dest_pts:array) -> array:
    avg_source, std_source = source_pts.mean(axis=1), source_pts.std(axis=1)
    avg_dest, std_dest = dest_pts.mean(axis=1), dest_pts.std(axis=1)

    N_source = np.array([[1/std_source[0], 0, 0], [0, 1/std_source[1], 0], [0, 0, 1]]) @ np.array([[1, 0, -avg_source[0]], [0, 1, -avg_source[1]], [0, 0, 1]])
    N_dest = np.array([[1/std_dest[0], 0, 0], [0, 1/std_dest[1], 0], [0, 0, 1]]) @ np.array([[1, 0, -avg_dest[0]], [0, 1, -avg_dest[1]], [0, 0, 1]])

    source_pts = N_source @ source_pts
    dest_pts = N_dest @ dest_pts
    zeros = np.zeros(source_pts.T.shape)
    
    A_upper = np.hstack([-source_pts.T, zeros, (dest_pts[0] * source_pts).T])
    A_lower = np.hstack([zeros, -source_pts.T, (dest_pts[1] * source_pts).T])
    A = np.vstack([A_upper, A_lower])
    V = np.linalg.svd(A)[2][-1]

    H = V.reshape(3, 3)
    return np.linalg.inv(N_dest) @ H @ N_source


if __name__ == "__main__":
    img = img_as_float(imread("tp4/data/pouliot.jpg"))
    H1 = np.array([[0.9752, 0.0013, -100.3164], [-0.4886, 1.7240, 24.8480], [-0.0016, 0.0004, 1.0000]])
    H2 = np.array([[0.1814, 0.7402, 34.3412], [1.0209, 0.1534, 60.3258], [0.0005, 0, 1.0000]])

    img_tf = appliqueTransformation(img, H2)

    plt.imshow(np.clip(img_tf, 0, 1))
    plt.axis('off')
    plt.show()
