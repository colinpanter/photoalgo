import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte
from skimage.io import imsave, imread

from stack import Stacker


IMAGES = {
    "pommange" : {'l': 'apple.jpeg', 'r': 'orange.jpeg'},
    "tree" : {'l': 'tree_summer.png', 'r': 'tree_winter.png'},
    "space" : {'l': 'spaceship.jpg', 'r': 'moon.jpg', 'mask': 'mask_spaceship.jpg'},
    "marsthedral" : {'l': 'cathedral.jpg', 'r': 'mars.jpg', 'mask': 'cathedral_mask.jpg'},
    "me" : {'l': 'assis.jpg', 'r': 'physicists.jpg', 'mask': 'assis_mask.jpg'},
    "holding_sun" : {'l': 'sun.jpg', 'r': 'holding.jpg', 'mask': 'sun_mask.jpg'},
    "irl_dbz" : {'l': 'spirit_bomb.jpg', 'r': 'posing.jpg', 'mask': 'spirit_bomb_mask.jpg'},
    "electroreseau" : {'l': 'electro.jpg', 'r': 'reseau.jpg'}
}


def normalize(img:np.ndarray) -> np.ndarray:
    return (img - img.min()) / (img.max() - img.min())


if __name__ == "__main__":
    name = "irl_dbz"
    img_l = img_as_float(imread(f"tp2/images/{IMAGES[name]['l']}"))
    img_r = img_as_float(imread(f"tp2/images/{IMAGES[name]['r']}"))

    if 'mask' in IMAGES[name]:
        mask = img_as_float(imread(f"tp2/images/{IMAGES[name]['mask']}"))
    else:
        mask = np.zeros(img_l.shape)
        mask[:, :img_l.shape[1]//2] = 1.

    n_filters = 5
    start = 1/4
    img_l_stack = Stacker(img_l, n_filters=n_filters, start=start).laplacian
    img_r_stack = Stacker(img_r, n_filters=n_filters, start=start).laplacian
    mask_stack = Stacker(mask, n_filters=n_filters, start=start/2).gaussian
    
    masked_l = img_l_stack * mask_stack
    masked_r = img_r_stack * (1 - mask_stack)

    for i in range(n_filters):
        ax_l = plt.subplot(n_filters, 3, 3*i+1)
        ax_l.imshow(normalize(masked_l[i]))
        ax_l.axis('off')

        ax_r = plt.subplot(n_filters, 3, 3*i+2)
        ax_r.imshow(normalize(masked_r[i]))
        ax_r.axis('off')

        ax_rl = plt.subplot(n_filters, 3, 3*i+3)
        ax_rl.imshow(normalize(masked_l[i] + masked_r[i]))
        ax_rl.axis('off')
    plt.tight_layout()
    plt.show()

    img = np.clip((img_l_stack * mask_stack + img_r_stack * (1 - mask_stack)).sum(axis=0), 0, 1)
    imsave(f"tp2/images/{name}.jpg", img_as_ubyte(img))
    plt.imshow(img, cmap='gray')

    plt.axis('off')
    plt.tight_layout()
    plt.show()