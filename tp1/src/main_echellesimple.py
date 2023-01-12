import numpy as np
import matplotlib.pyplot as plt

from reconstructor import Reconstructor


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


def align(r:np.ndarray, g:np.ndarray, b:np.ndarray) -> np.ndarray:
    pass


if __name__=="__main__":
    # fig = plt.figure(figsize=(16,4))
    fig = plt.figure(figsize=(8,8))

    img = plt.imread("tp1/images/00911v.jpg")
    h = img.shape[0] // 3
    b, g, r = img[:h], img[h:-h], img[-h:]

    height = min((x.shape[0] for x in [r, g, b]))
    r, g, b = r[:height], g[:height], b[:height]

    reconstructor = Reconstructor(r, g, b)

    # ax = plt.subplot(141)
    # imshow(r, ax, "Red channel")
    
    # ax = plt.subplot(142)
    # imshow(g, ax, "Green channel")
    
    # ax = plt.subplot(143)
    # imshow(b, ax, "Blue channel")

    # ax = plt.subplot(144)
    ax = plt.subplot(111)
    imshow(reconstructor.superpose(red_shift=(-15, 0), green_shift=(-1, 0)), ax, "Reconstructed image")

    plt.show()
