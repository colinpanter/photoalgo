import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from skimage import img_as_ubyte, img_as_float
from pathlib import Path

from splicer import Splicer
from splitter import Splitter
from optimizers import Optimizer


def reconstruct(img_path:Path, optimizer:Optimizer, show=False):
    # Read image
    img = img_as_float(imread(img_path))

    name = img_path.name.split(sep='.')[0] + '.jpg'
    root_dir = img_path.parent.parent

    reconstruction = root_dir.joinpath("reconstructions", name)
    comparison = root_dir.joinpath("comparisons", name)

    reconstruction.parent.mkdir(exist_ok=True)
    comparison.parent.mkdir(exist_ok=True)

    # Split image in RGB channels
    splitter = Splitter()
    r, g, b = splitter.split(img, crop=0.1)
    splicer = Splicer(r, g, b)

    # Calculate optimal translations

    red_shift, green_shift = optimizer.optimize(splicer)

    # Display original - RGB channels - Reconstruction
    splitter = Splitter()
    r, g, b = splitter.split(img)

    splicer = Splicer(r, g, b)
    spliced_img = splicer.splice(red_shift=red_shift, green_shift=green_shift)

    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(3, 5)
    ax_original = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[2, 1])
    ax_g = fig.add_subplot(gs[1, 1])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_splice = fig.add_subplot(gs[:, 2:])

    ax_original.imshow(img, cmap="gray")
    ax_original.axis('off')

    ax_r.imshow(r, cmap="Reds_r")
    ax_g.imshow(g, cmap="Greens_r")
    ax_b.imshow(b, cmap="Blues_r")
    ax_r.axis('off')
    ax_g.axis('off')
    ax_b.axis('off')

    ax_splice.imshow(spliced_img)
    ax_splice.axis('off')

    plt.axis('off')
    imsave(reconstruction, img_as_ubyte(spliced_img))
    plt.savefig(comparison)
    if show:
        plt.show()
    plt.clf
