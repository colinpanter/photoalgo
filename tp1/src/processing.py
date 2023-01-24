import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_ubyte, img_as_float
import matplotlib.pyplot as plt


if __name__ == "__main__":
    name = "blocks"
    r = img_as_float(imread(f"tp1/images/temp/{name}_r.tiff"))
    g = img_as_float(imread(f"tp1/images/temp/{name}_g.tiff"))
    b = img_as_float(imread(f"tp1/images/temp/{name}_b.tiff"))

    plt.figure(figsize=(16, 8))

    # plt.subplot(131).imshow(r, cmap="Greys_r")
    # plt.subplot(132).imshow(g, cmap="Greys_r")
    # plt.subplot(133).imshow(b, cmap="Greys_r")

    # print(np.mean(r), np.mean(g), np.mean(b))
    # print(r[-1, 0], g[-1, 0], b[-1, 0])

    r_max, b_max, g_max = 0.573006790264744, 0.4758373388265812, 0.4587472343022812

    r = img_as_ubyte(np.clip(r / r_max, 0, 1))
    g = img_as_ubyte(np.clip(g / g_max, 0, 1))
    b = img_as_ubyte(np.clip(b / b_max, 0, 1))

    imsave(f"tp1/images/custom/{name}_r.jpg", r)
    imsave(f"tp1/images/custom/{name}_g.jpg", g)
    imsave(f"tp1/images/custom/{name}_b.jpg", b)

    # img = np.stack([r, g, b], axis=2)
    # plt.imshow(img)

    # plt.show()

