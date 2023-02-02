import numpy as np
import matplotlib.pyplot as plt

from filters import GaussianFilter, AntiGaussianFilter, GaussianSharpeningFilter, HighPassFilter, LowPassFilter
from align_images import align_images


if __name__ == "__main__":
    img = plt.imread("tp2/hybrid_python/Albert_Einstein.png")[:, :, 0]

    plt.figure(figsize=(16, 8))

    filter = LowPassFilter(10)

    ax = plt.subplot(111)
    ax.imshow(filter(img), cmap="gray")
    ax.axis('off')

    plt.tight_layout()
    plt.show()
