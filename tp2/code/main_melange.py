import numpy as np
import matplotlib.pyplot as plt

from stack import Stacker


if __name__ == "__main__":
    img_l = plt.imread("tp2/images/orange.jpeg") / 255#.sum(axis=2) / 3
    img_r = plt.imread("tp2/images/apple.jpeg") / 255#.sum(axis=2) / 3

    mask = np.zeros(img_l.shape)
    mask[:, img_l.shape[1]//2:] = 1.

    n_filters = 3
    img_l_stack = Stacker(img_l, n_filters=n_filters).laplacian
    img_r_stack = Stacker(img_r, n_filters=n_filters).laplacian
    mask_stack = Stacker(mask, n_filters=n_filters).gaussian
    
    img = (img_l_stack * mask_stack + img_r_stack * (1 - mask_stack)).sum(axis=0)
    
    plt.imshow(img, cmap='gray')

    plt.axis('off')
    plt.tight_layout()
    plt.show()