import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave

from homography import appliqueTransformation


if __name__ == "__main__":
    img = img_as_float(imread("tp4/data/pouliot.jpg"))
    H1 = np.array([[0.9752, 0.0013, -100.3164], [-0.4886, 1.7240, 24.8480], [-0.0016, 0.0004, 1.0000]])
    H2 = np.array([[0.1814, 0.7402, 34.3412], [1.0209, 0.1534, 60.3258], [0.0005, 0, 1.0000]])

    img_tf = appliqueTransformation(img, H2)

    plt.imshow(np.clip(img_tf, 0, 1))
    plt.axis('off')
    plt.show()
