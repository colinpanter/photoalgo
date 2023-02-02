import numpy as np
import matplotlib.pyplot as plt

from filters import GaussianFilter, AntiGaussianFilter, GaussianSharpeningFilter, HighPassFilter, LowPassFilter
from align_images import align_images


if __name__ == "__main__":
    img_hf = plt.imread("tp2/hybrid_python/Albert_Einstein.png")[:, :, 0]
    img_lf = plt.imread("tp2/hybrid_python/Marilyn_Monroe.png")[:, :, 0]

    sigma = 5
    cutoff = 5

    img_hf, img_lf = align_images(img_hf, img_lf)

    plt.figure(figsize=(16, 8))

    filter_hf = GaussianFilter(sigma)
    filtered_hf = filter_hf(img_hf)
    # ax = plt.subplot(131)
    # ax.imshow(filtered_hf, cmap="gray")
    # ax.axis('off')
    
    filter_lf = AntiGaussianFilter(sigma+2)
    filtered_lf = filter_lf(img_lf)
    # ax = plt.subplot(132)
    # ax.imshow(filtered_lf, cmap="gray")
    # ax.axis('off')

    # ax = plt.subplot(133)
    ax = plt.subplot(111)
    ax.imshow(filtered_lf + filtered_hf, cmap="gray")
    ax.axis('off')

    plt.tight_layout()
    plt.show()
