import numpy as np
import matplotlib.pyplot as plt

from filters import GaussianFilter, AntiGaussianFilter, GaussianSharpeningFilter


if __name__ == "__main__":
    img = plt.imread("tp2/images/Albert_Einstein.png")[:, :, 0]

    plt.figure(figsize=(12, 8))

    # ax = plt.subplot(141)
    # ax.imshow(img, cmap="gray")
    # ax.set_title(f"Original")
    # ax.axis('off')
    
    for i in range(3):
        sigma = (i + 1) * 5
        filter = AntiGaussianFilter(sigma)
        img_filtered = filter(img)

        for j, alpha in enumerate([0, 1/2, 1, 3/2, 2]):
            ax = plt.subplot(3, 5, j+1 + 5*i)
            ax.imshow(np.clip(img + alpha * img_filtered, 0, 1), cmap="gray")

            ax.set_title(f"$\sigma = {sigma}, \\alpha = {alpha}$")
            ax.axis('off')

    plt.tight_layout()
    plt.show()
