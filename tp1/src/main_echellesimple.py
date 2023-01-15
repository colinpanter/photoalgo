import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from splicer import Splicer


def imshow(img:np.ndarray, ax:plt.Axes, title) -> None:
    ax.imshow(img, cmap="gray")
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)
    ax.set_title(title)


def show_img(img_path):
    img = plt.imread(img_path)
    h = img.shape[0] // 3
    b, g, r = img[:h], img[h:-h], img[-h:]

    height = min((x.shape[0] for x in [r, g, b]))
    r, g, b = r[:height], g[:height], b[:height]

    splicer = Splicer(r, g, b)

    red_shift, green_shift = splicer.optimize()

    # _ = plt.figure(figsize=(12,12))

    # ax = plt.subplot(221)
    # imshow(r, ax, "Red channel")
    
    # ax = plt.subplot(222)
    # imshow(g, ax, "Green channel")
    
    # ax = plt.subplot(223)
    # imshow(b, ax, "Blue channel")

    # ax = plt.subplot(224)

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111)

    imshow(splicer.splice(red_shift=red_shift, green_shift=green_shift), ax, "Reconstructed image")

    plt.show()


if __name__=="__main__":
    img_dir = Path("tp1/images")
    for img in img_dir.glob("*.jpg"):
        show_img(img)