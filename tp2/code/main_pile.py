import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread

from stack import Stacker


if __name__ == "__main__":
    img = img_as_float(imread("tp2/images/colympic.jpg"))

    plt.figure(figsize=(16,6))

    stack = Stacker(img, n_filters=5, start=1/4)
    length = stack.gaussian.shape[0]
    for i in range(length):
        ax = plt.subplot(2, length, i+1)
        ax.imshow(stack.gaussian[i], cmap="gray")
        ax.set_title("Pile gaussienne" if i == 0 else None)
        ax.axis('off')
        
        ax = plt.subplot(2, length, length+i+1)
        l = np.clip(np.abs(stack.laplacian[i]), 0, 1)
        ax.imshow(l / l.max(), cmap="gray")
        ax.set_title("Pile laplacienne" if i == 0 else None)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
