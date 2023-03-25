import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io, img_as_float, img_as_ubyte

from descriptors import generate_descriptors, find_corners, match_features, RANSAC
from homography import calculate_homography, appliqueTransformation


def display_points(img1, img2, pts1, pts2, name):
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(121)
    ax.imshow(img1)
    ax.scatter(pts1[0], pts1[1], 4, 'r')
    ax.axis('off')

    ax = plt.subplot(122)
    ax.imshow(img2)
    ax.scatter(pts2[0], pts2[1], 4, 'r')
    ax.axis('off')

    plt.tight_layout()

    plt.savefig(f"tp4/web/images/automatique/etapes/{name}.jpg", bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    imgpath1 = "tp4/data/2-PartieAutomatique/Serie2/IMG_2415.JPG"
    img1 = img_as_float(io.imread(imgpath1))
    img_gray1 = color.rgb2gray(io.imread(imgpath1))
 
    imgpath2 = "tp4/data/2-PartieAutomatique/Serie2/IMG_2416.JPG"
    img2 = img_as_float(io.imread(imgpath2))
    img_gray2 = color.rgb2gray(io.imread(imgpath2))
    
    print("Finding corners of image 1")
    pts1 = find_corners(img_gray1)
    print("Finding corners of image 2")
    pts2 = find_corners(img_gray2)

    display_points(img1, img2, pts1.T, pts2.T, "all_corners")

    print("Finding corners of image 1")
    pts1 = find_corners(img_gray1, 500)
    print("Finding corners of image 2")
    pts2 = find_corners(img_gray2, 500)

    display_points(img1, img2, pts1.T, pts2.T, "corners")

    print("Extracting features of image 1")
    desc1, pts1 = generate_descriptors(img_gray1, pts1)
    print("Extracting features of image 2")
    desc2, pts2 = generate_descriptors(img_gray2, pts2)

    print("Matching features")
    correspondance = match_features(desc1, desc2)

    pts1 = pts1[correspondance[0]].T
    pts2 = pts2[correspondance[1]].T
    
    display_points(img1, img2, pts1, pts2, "matching")

    pts1 = np.vstack([pts1, [[1]* pts1.shape[1]]])
    pts2 = np.vstack([pts2, [[1]* pts2.shape[1]]])

    print("RANSAC")
    pts1, pts2 = RANSAC(pts1, pts2)
    
    display_points(img1, img2, pts1, pts2, "RANSAC")

    # H = calculate_homography(pts2, pts1)
    # img_tf, lim = appliqueTransformation(img2, H, True)
    
    total_lim = (0, img1.shape[1], 0, img1.shape[0])
    pairs = [(img1, total_lim)]

    tf_data = [(img2, pts2, pts1)]
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
