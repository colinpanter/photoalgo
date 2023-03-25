import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte


from homography import calculate_homography, appliqueTransformation
from utils import read_points


DATA_DIR = Path("tp4/data/1-PartieManuelle/")


if __name__ == "__main__":
    serie, img_names = "Serie1", ["IMG_2415.JPG", "IMG_2416.JPG", "IMG_2417.JPG"]
    # serie, img_names = "Serie2", ["IMG_2425.JPG", "IMG_2426.JPG", "IMG_2427.JPG"]
    # serie, img_names = "Serie3", ["IMG_2409.JPG", "IMG_2410.JPG", "IMG_2411.JPG"]

    pts_cl_l = read_points(DATA_DIR.joinpath(serie, "pts", "pts1_12.txt"), homogenous=True).T
    pts_cl_c = read_points(DATA_DIR.joinpath(serie, "pts", "pts2_12.txt"), homogenous=True).T

    pts_cr_r = read_points(DATA_DIR.joinpath(serie, "pts", "pts3_32.txt"), homogenous=True).T
    pts_cr_c = read_points(DATA_DIR.joinpath(serie, "pts", "pts2_32.txt"), homogenous=True).T

    img_left = img_as_float(imread(DATA_DIR.joinpath(serie, img_names[0])))
    img_center = img_as_float(imread(DATA_DIR.joinpath(serie, img_names[1])))
    img_right = img_as_float(imread(DATA_DIR.joinpath(serie, img_names[2])))


    total_lim = (0, img_center.shape[1], 0, img_center.shape[0])
    pairs = [(img_center, total_lim)]

    tf_data = [(img_left, pts_cl_l, pts_cl_c), (img_right, pts_cr_r, pts_cr_c)]
    for i, (img, pts_img, pts_ref) in enumerate(tf_data):
        H = calculate_homography(pts_img, pts_ref)
        img_tf, lim = appliqueTransformation(img, H, True)

        total_lim = (min(total_lim[0], lim[0]), max(total_lim[1], lim[1]), min(total_lim[2], lim[2]), max(total_lim[3], lim[3]))
        pairs.append((img_tf, lim))
        
        plt.figure(figsize=(12, 12))
        
        ax = plt.subplot(121)
        ax.imshow(img_center)
        ax.scatter(pts_ref[0], pts_ref[1], 4, 'r')
        ax.axis('off')

        ax = plt.subplot(122)
        ax.imshow(img)
        ax.scatter(pts_img[0], pts_img[1], 4, 'r')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"tp4/web/images/manuelle/{serie}/correspondance_{i}.jpg", bbox_inches='tight')
        plt.clf()
    
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

    imsave(f"tp4/web/images/manuelle/{serie}.jpg", img_as_ubyte(np.clip(total_img, 0, 1)))
