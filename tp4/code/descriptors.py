import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, color
from skimage import io
from scipy.signal import convolve2d as conv2
from scipy.ndimage.filters import generic_filter as gf
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

from homography import calculate_homography


def fspecial_gaussian(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    source: https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def find_corners(img, n=None):
    threshold = 1e-3

    g1 = fspecial_gaussian([9, 9], 1)  # Gaussian with sigma_d
    g2 = fspecial_gaussian([11, 11], 1.5)  # Gaussian with sigma_i

    img1 = conv2(img, g1, 'same')  # blur image with sigma_d
    Ix = conv2(img1, np.array([[-1, 0, 1]]), 'same')  # take x derivative
    Iy = conv2(img1, np.transpose(np.array([[-1, 0, 1]])), 'same')  # take y derivative

    # Compute elements of the Harris matrix H
    # we can use blur instead of the summing window
    Ix2 = conv2(np.multiply(Ix, Ix), g2, 'same')
    Iy2 = conv2(np.multiply(Iy, Iy), g2, 'same')
    IxIy = conv2(np.multiply(Ix, Iy), g2, 'same')
    eps = 2.2204e-16
    R = np.divide(np.multiply(Ix2, Iy2) - np.multiply(IxIy, IxIy),(Ix2 + Iy2 + eps))

    # don't want corners close to image border
    R[0:15] = 0  # all columns from the first 15 lines
    R[-16:] = 0  # all columns from the last 15 lines
    R[:, 0:15] = 0  # all lines from the first 15 columns
    R[:, -16:] = 0  # all lines from the last 15 columns

    # non-maxima suppression within 3x3 windows
    Rmax = gf(R, np.max, footprint=np.ones((3, 3)))
    Rmax[Rmax != R] = 0  # suppress non-max
    Rmax[Rmax < threshold] = 0
    v = Rmax[Rmax != 0]
    y, x = np.nonzero(Rmax)
    
    p = np.vstack([x, y]).T
    if n is not None:
        # TODO : Adaptive Non-Maximal Suppression
        choice = np.random.choice(p.shape[0], n)
        p = p[choice]

    return p


def generate_descriptors(img, p):
    # Descriptors positions (9x9 spaced by 5 pixels)
    dx, dy = np.meshgrid(5*np.arange(-4, 5), 5*np.arange(-4, 5))
    desc_x = p[:, 0].reshape((-1, 1, 1)) + dx
    desc_y = p[:, 1].reshape((-1, 1, 1)) + dy
    desc_x = np.clip(desc_x, 0, img.shape[1]-1) # Out of bounds
    desc_y = np.clip(desc_y, 0, img.shape[0]-1) # Out of bounds

    blurred_img = gaussian_filter(img, 2) # Anti-Aliasing

    # Normalization
    desc: np.ndarray = blurred_img[desc_y, desc_x]
    mu = desc.mean(axis=(1,2), keepdims=True)
    sigma = desc.std(axis=(1,2), keepdims=True)
    desc = (desc-mu) / sigma

    return desc, p


def match_features(desc1, desc2):
    distance = cdist(desc1.reshape((desc1.shape[0], -1)), desc2.reshape((desc2.shape[0], -1)), 'euclidean')
    best_desc1, best_desc2 = [], []

    threshold = .6
    prev = 0
    while prev < threshold and len(best_desc1) < len(desc1) and len(best_desc2) < len(desc2):
        y = np.arange(distance.shape[0])
        distance_sorted = distance.argsort(axis=1)

        best_desc1.append((distance[y, distance_sorted[:, 0]] / distance[y, distance_sorted[:, 1]]).argmin())
        best_desc2.append(distance_sorted[best_desc1[-1], 0])
        prev = (distance[y, distance_sorted[:, 0]] / distance[y, distance_sorted[:, 1]]).min()

        distance[:, best_desc2[-1]] = np.inf

    return np.array(best_desc1), np.array(best_desc2)

def RANSAC(pts1, pts2):
    threshold = 16
    consistants = []
    for _ in range(100):
        choice = np.random.choice(pts1.shape[1], 4)
        H = calculate_homography(pts1[:, choice], pts2[:, choice])
        
        pts_tf = H @ pts1
        pts_tf = pts_tf / pts_tf[2]
        
        error = np.square(pts_tf - pts2).sum(axis=0)
        consistants.append(error < threshold)

    best = max(consistants, key=lambda x : x.sum())

    return pts1[:, best], pts2[:, best]
