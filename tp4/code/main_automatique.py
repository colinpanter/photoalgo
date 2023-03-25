import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io, img_as_float, img_as_ubyte

from descriptors import generate_descriptors, find_corners, match_features, RANSAC
from homography import calculate_homography, appliqueTransformation
from sets import automatic


if __name__ == "__main__":
    name = "Serie3"
    data = automatic[name]

    total_lim = (0, 0, 0, 0)
    pairs = []

    # Reference image
    datum = data[0]
    datum["img"] = img_as_float(io.imread(datum["path"]))
    datum["gray"] = color.rgb2gray(io.imread(datum["path"]))

    corners = find_corners(datum["gray"], 500)
    datum["descriptors"], datum["pts"] = generate_descriptors(datum["gray"], corners)

    datum["H"] = np.identity(3)

    img_tf, lim = appliqueTransformation(datum["img"], datum["H"], True)

    total_lim = (min(total_lim[0], lim[0]), max(total_lim[1], lim[1]), min(total_lim[2], lim[2]), max(total_lim[3], lim[3]))
    pairs.append((img_tf, lim))

    for i, datum in enumerate(data[1:]):
        print(i)
        datum["img"] = img_as_float(io.imread(datum["path"]))
        datum["gray"] = color.rgb2gray(io.imread(datum["path"])) if len(datum["img"].shape) == 3 else datum["img"]

        corners = find_corners(datum["gray"], 500)
        datum["descriptors"], datum["pts"] = generate_descriptors(datum["gray"], corners)

        pts_source = data[datum["ref"]]["pts"]
        desc_source = data[datum["ref"]]["descriptors"]

        correspondance = match_features(desc_source, datum["descriptors"])
        pts_source = pts_source[correspondance[0]].T
        pts_dest = datum["pts"][correspondance[1]].T

        pts_source = np.vstack([pts_source, [[1]* pts_source.shape[1]]])
        pts_dest = np.vstack([pts_dest, [[1]* pts_dest.shape[1]]])

        pts_source, pts_dest = RANSAC(pts_source, pts_dest)
        
        plt.figure(figsize=(12, 12))
        
        ax = plt.subplot(121)
        ax.imshow(data[datum["ref"]]["img"])
        ax.scatter(pts_source[0], pts_source[1], 4, 'r')
        ax.axis('off')

        ax = plt.subplot(122)
        ax.imshow(datum["img"])
        ax.scatter(pts_dest[0], pts_dest[1], 4, 'r')
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(f"tp4/web/images/automatique/{name}/correspondance_{i}.jpg", bbox_inches='tight')
        plt.clf()

        H = calculate_homography(pts_dest, pts_source)
        datum["H"] = data[datum["ref"]]["H"] @ H

        img_tf, lim = appliqueTransformation(datum["img"], datum["H"], True)

        total_lim = (min(total_lim[0], lim[0]), max(total_lim[1], lim[1]), min(total_lim[2], lim[2]), max(total_lim[3], lim[3]))
        pairs.append((img_tf, lim))
    
    shape = (total_lim[3] - total_lim[2], total_lim[1] - total_lim[0], 3)
    total_img = np.zeros(shape, dtype=float)
    count = np.zeros(shape[:-1])
    
    for img, lim in pairs:
        slice_x = slice(lim[0] - total_lim[0], lim[0] - total_lim[0] + img.shape[1])
        slice_y = slice(lim[2] - total_lim[2], lim[2] - total_lim[2] + img.shape[0])

        total_img[slice_y, slice_x] += img
        count[slice_y, slice_x][img.sum(axis=-1) > 0] += 1
    total_img[count > 0] = (total_img[count > 0].T / count[count > 0]).T

    io.imsave(f"tp4/web/images/automatique/{name}.jpg", img_as_ubyte(np.clip(total_img, 0, 1)))
