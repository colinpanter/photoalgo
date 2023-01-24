import matplotlib.pyplot as plt
from pathlib import Path

from splicer import GorskiiSplicer
from metrics import MSE
from optimizers import MultiScaleOptimizer
from reconstruction import reconstruct


if __name__=="__main__":
    metric = MSE()
    optimizer = MultiScaleOptimizer(metric)

    img_dir = Path("tp1/images/sergei")
    images = list(img_dir.glob("*.tif"))

    save_dir = Path("tp1/web/img/highres")
    for i, img in enumerate(images):
        print(f"Image {i+1}/{len(images)}", end="\r")

        splicer = GorskiiSplicer(img)
        reconstruct(splicer, optimizer, img.name.split('.')[0]+'.jpg', save_dir, show=False)
    plt.show()
