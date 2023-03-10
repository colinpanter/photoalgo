import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float


from homography import calculate_homography, appliqueTransformation
from utils import read_points


if __name__ == "__main__":
    pts_cl_l = read_points("tp4/data/1-PartieManuelle/Serie1/pts_serie1/pts1_12.txt", homogenous=True).T
    pts_cl_c = read_points("tp4/data/1-PartieManuelle/Serie1/pts_serie1/pts2_12.txt", homogenous=True).T

    pts_cr_r = read_points("tp4/data/1-PartieManuelle/Serie1/pts_serie1/pts3_32.txt", homogenous=True).T
    pts_cr_c = read_points("tp4/data/1-PartieManuelle/Serie1/pts_serie1/pts2_32.txt", homogenous=True).T

    img_left = img_as_float(imread("tp4/data/1-PartieManuelle/Serie1/IMG_2415.JPG"))
    img_center = img_as_float(imread("tp4/data/1-PartieManuelle/Serie1/IMG_2416.JPG"))
    img_right = img_as_float(imread("tp4/data/1-PartieManuelle/Serie1/IMG_2417.JPG"))


    total_lim = (0, img_center.shape[1], 0, img_center.shape[0])
    pairs = [(img_center, total_lim)]

    tf_data = [(img_left, pts_cl_l, pts_cl_c), (img_right, pts_cr_r, pts_cr_c)]
    for img, pts_img, pts_ref in tf_data:
        H = calculate_homography(pts_img, pts_ref)
        img_tf, lim = appliqueTransformation(img, H, True)

        total_lim = (min(total_lim[0], lim[0]), max(total_lim[1], lim[1]), min(total_lim[2], lim[2]), max(total_lim[3], lim[3]))
        pairs.append((img_tf, lim))
    
    shape = (total_lim[3] - total_lim[2], total_lim[1] - total_lim[0], 3)
    total_img = np.zeros(shape, dtype=float)
    count = np.zeros(shape[:-1])
    
    for img, lim in pairs:
        slice_x = slice(lim[0] - total_lim[0], lim[0] - total_lim[0] + img.shape[1])
        slice_y = slice(lim[2] - total_lim[2], lim[2] - total_lim[2] + img.shape[0])

        # total_img[slice_y, slice_x] = np.maximum(total_img[slice_y, slice_x], img)

        total_img[slice_y, slice_x] += img
        count[slice_y, slice_x][img.sum(axis=-1) > 0] += 1
    total_img[count > 0] = (total_img[count > 0].T / count[count > 0]).T

    plt.imshow(np.clip(total_img, 0, 1))
    plt.show()
