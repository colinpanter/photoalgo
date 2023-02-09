import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte

from stack import Stacker


if __name__ == "__main__":
    img = img_as_float(plt.imread("tp2/images/colympic.jpg"))

    plt.figure(figsize=(16,6))

    stack = Stacker(img, n_filters=5, start=1/2)
    length = stack.gaussian.shape[0]
    for i in range(length):
        ax = plt.subplot(2, length, i+1)
        ax.imshow(stack.gaussian[i], cmap="gray")
        ax.axis('off')
        
        ax = plt.subplot(2, length, length+i+1)
        l = stack.laplacian[i]
        ax.imshow((l-l.min()) / (l.max()-l.min()), cmap="gray")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
