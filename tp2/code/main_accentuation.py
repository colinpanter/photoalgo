import numpy as np
import matplotlib.pyplot as plt

from filters import GaussianFilter, AntiGaussianFilter, GaussianSharpeningFilter


if __name__ == "__main__":
    img = plt.imread("tp2/hybrid_python/Albert_Einstein.png")[:, :, 0]

    plt.figure(figsize=(16, 8))

    ax = plt.subplot(141)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Original")
    ax.axis('off')
    
    for i in range(3):
        sigma = (i + 1) * 50
        filter = GaussianSharpeningFilter(sigma)
        img_filtered = filter(img)

        ax = plt.subplot(140 + i + 2)
        ax.imshow(img_filtered, cmap="gray")

        ax.set_title(f"$\sigma = {sigma}$")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
