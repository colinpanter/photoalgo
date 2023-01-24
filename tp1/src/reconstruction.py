import matplotlib.pyplot as plt
from skimage.io import imsave, imread
from skimage import img_as_ubyte, img_as_float
from pathlib import Path

from splicer import Splicer
from splitter import Splitter
from optimizers import Optimizer


def reconstruct(splicer:Splicer, optimizer:Optimizer, name:str, save_dir:Path, show=False):
    # Calculate optimal translations
    red_shift, green_shift = optimizer.optimize(splicer)
    print('{' + f'"img": "{name}", "rx": {-red_shift[1]}, "ry": {-red_shift[0]}, "gx": {-green_shift[1]}, "gy": {-green_shift[0]}' + '}')
    display(splicer, red_shift, green_shift, name, save_dir, show)


def display(splicer:Splicer, red_shift:tuple[int,int], green_shift:tuple[int,int], name:str, save_dir:Path, show=False):
    reconstruction = save_dir.joinpath("reconstructions", name)
    comparison = save_dir.joinpath("comparisons", name)

    reconstruction.parent.mkdir(exist_ok=True)
    comparison.parent.mkdir(exist_ok=True)

    # Display original - RGB channels - Reconstruction
    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(3, 5)
    ax_original = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[2, 1])
    ax_g = fig.add_subplot(gs[1, 1])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_splice = fig.add_subplot(gs[:, 2:])

    ax_original.imshow(splicer.get_original(), cmap="gray")
    ax_original.axis('off')

    r, g, b = splicer.split()
    ax_r.imshow(r, cmap="Reds_r")
    ax_g.imshow(g, cmap="Greens_r")
    ax_b.imshow(b, cmap="Blues_r")
    ax_r.axis('off')
    ax_g.axis('off')
    ax_b.axis('off')
    
    spliced_img = splicer.splice(red_shift=red_shift, green_shift=green_shift)
    ax_splice.imshow(spliced_img)
    ax_splice.axis('off')

    plt.axis('off')
    plt.tight_layout()
    imsave(reconstruction, img_as_ubyte(spliced_img))
    plt.savefig(comparison)
    if show:
        plt.show()
    plt.clf
