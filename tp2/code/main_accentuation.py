import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float

from filters import AntiGaussianFilter


IMAGES = {
    "Einstein": "Albert_Einstein.png",
    "cat": "cat.jpg",
    }


if __name__ == "__main__":
    name = "Einstein"
    img = img_as_float(imread(f"tp2/images/{IMAGES[name]}"))

    plt.figure(figsize=(12, 8))
    
    alphas = [0, 0.5, 1., 2]
    sigma = 20
    
    for i, alpha in enumerate(alphas):
        filter = AntiGaussianFilter(sigma)
        img_filtered = filter(img)

        ax = plt.subplot(1, len(alphas), i+1)
        ax.imshow(np.clip(img + alpha * img_filtered, 0, 1), cmap="gray")

        ax.set_title(f"$\sigma = {sigma}, \\alpha = {alpha}$")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"tp2/images/{name}_sharpened.png", bbox_inches='tight')
    plt.show()
