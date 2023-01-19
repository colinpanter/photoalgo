import matplotlib.pyplot as plt
from pathlib import Path

from metrics import MSE
from optimizers import SingleScaleOptimizer
from reconstruction import reconstruct


if __name__=="__main__":
    metric = MSE()
    optimizer = SingleScaleOptimizer(metric)

    img_dir = Path("tp1/images/sergei")
    images = list(img_dir.glob("*.jpg"))
    for i, img in enumerate(images):
        print(f"Image {i+1}/{len(images)}", end="\r")
        reconstruct(img, optimizer)
