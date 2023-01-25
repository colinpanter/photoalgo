import matplotlib.pyplot as plt
from pathlib import Path

from splicer import ThreeBWSplicer
from metrics import MSE, EdgeMSE
from optimizers import SingleScaleOptimizer
from reconstruction import reconstruct


if __name__=="__main__":
    names = ["plush", "blocks"]

    metric = MSE()
    optimizer = SingleScaleOptimizer(metric, search_range=8)

    save_dir = Path("tp1/web/img/custom")
    for i, name in enumerate(names):
        print(f"Image {i+1}/{len(names)}", end="\r")
        r = Path(f"tp1/images/custom/{name}_r.jpg")
        g = Path(f"tp1/images/custom/{name}_g.jpg")
        b = Path(f"tp1/images/custom/{name}_b.jpg")

        splicer = ThreeBWSplicer(r, g, b)
        reconstruct(splicer, optimizer, name+".jpg", save_dir, show=False)
