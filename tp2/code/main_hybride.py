import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte

from filters import GaussianFilter, AntiGaussianFilter
from align_images import align_images


IMAGES = {
    "Einroe": {'lf': "Albert_Einstein.png", 'hf': "Marilyn_Monroe.png"},
    "trucktimus" : {'lf': 'truck.jpg', 'hf': 'optimus.jpg'},
    "catdog": {'lf': 'dog.jpg', 'hf': 'cat.jpg'},
    "colympic": {'lf': 'olympic.jpg', 'hf': 'colosseo.jpg'}
    }


def get_freq(img:np.ndarray) -> np.ndarray:
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))


if __name__ == "__main__":
    name = "colympic"
    img_hf = img_as_float(imread(f"tp2/images/{IMAGES[name]['hf']}", as_gray=True))
    img_lf = img_as_float(imread(f"tp2/images/{IMAGES[name]['lf']}", as_gray=True))


    img_hf, img_lf = align_images(img_hf, img_lf)

    plt.figure(figsize=(16, 8))

    sigma = 5
    filter_highpass = AntiGaussianFilter(7)
    filtered_hf = filter_highpass(img_hf)
    
    filter_lowpass = GaussianFilter(3)
    filtered_lf = filter_lowpass(img_lf)

    img = filtered_lf + filtered_hf

    imsave(f"tp2/images/{name}.jpg", img_as_ubyte(np.clip(img, 0, 1)))
    plt.imshow(np.clip(img, 0, 1), cmap="gray")

    plt.axis('off')
    plt.tight_layout()
    plt.show()

    ax = plt.subplot(4, 4, 1)
    ax.imshow(img_lf, cmap='gray')
    ax.set_title("Image originale\n(basses fréquences)")
    ax.axis('off')

    ax = plt.subplot(4, 4, 2)
    ax.imshow(filtered_lf, cmap='gray')
    ax.set_title("Image filtrée\n(basses fréquences)")
    ax.axis('off')

    ax = plt.subplot(4, 4, 5)
    ax.imshow(get_freq(img_lf), cmap='gray')
    ax.set_title("Fréquences originales\n(basses fréquences)")
    ax.axis('off')

    ax = plt.subplot(4, 4, 6)
    ax.imshow(get_freq(filtered_lf), cmap='gray')
    ax.set_title("Fréquences filtrées\n(basses fréquences)")
    ax.axis('off')

    ax = plt.subplot(4, 4, 9)
    ax.imshow(img_hf, cmap='gray')
    ax.set_title("Image originale (hautes\nfréquences)")
    ax.axis('off')

    ax = plt.subplot(4, 4, 10)
    ax.imshow(filtered_hf, cmap='gray')
    ax.set_title("Image filtrée (hautes\nfréquences)")
    ax.axis('off')

    ax = plt.subplot(4, 4, 13)
    ax.imshow(get_freq(img_hf), cmap='gray')
    ax.set_title("Fréquences originales\n(hautes fréquences)")
    ax.axis('off')

    ax = plt.subplot(4, 4, 14)
    ax.imshow(get_freq(filtered_hf), cmap='gray')
    ax.set_title("Fréquences filtrées\n(hautes fréquences)")
    ax.axis('off')

    ax = plt.subplot(2, 2, 2)
    ax.imshow(np.clip(img, 0, 1), cmap='gray')
    ax.set_title("Image hybride")
    ax.axis('off')

    ax = plt.subplot(2, 2, 4)
    ax.imshow(get_freq(img), cmap='gray')
    ax.set_title("Fréquences hybrides")
    ax.axis('off')

    plt.tight_layout()
    plt.show()
