import matplotlib.pyplot as plt
from pathlib import Path

from splicer import ThreeRGBSplicer
from metrics import MSE, EdgeMSE
from optimizers import MultiScaleOptimizer
from reconstruction import reconstruct


if __name__=="__main__":
    names = ["cats", "light", "outside", "candles", "tv"]
    # names = ["light"]

    metric = MSE()
    optimizer = MultiScaleOptimizer(metric)

    save_dir = Path("tp1/web/img/custom")
    for i, name in enumerate(names):
        print(f"Image {i+1}/{len(names)}", end="\r")
        r = Path(f"tp1/images/custom/{name}_r.jpg")
        g = Path(f"tp1/images/custom/{name}_g.jpg")
        b = Path(f"tp1/images/custom/{name}_b.jpg")

        splicer = ThreeRGBSplicer(r, g, b)
        reconstruct(splicer, optimizer, name+".jpg", save_dir, show=False)
    
    # r = Path(f"tp1/images/custom/light_r.jpg")
    # g = Path(f"tp1/images/custom/light_g.jpg")
    # b = Path(f"tp1/images/custom/light_b.jpg")
    # reconstruct(ThreeRGBSplicer(b, r, r), optimizer, "light_brr.jpg", save_dir)
    # reconstruct(ThreeRGBSplicer(r, r, r), optimizer, "light_rrr.jpg", save_dir)
    # reconstruct(ThreeRGBSplicer(b, g, r), optimizer, "light_bgr.jpg", save_dir)
    # reconstruct(ThreeRGBSplicer(b, r, b), optimizer, "light_brb.jpg", save_dir)
    # reconstruct(ThreeRGBSplicer(r, g, r), optimizer, "light_rgr.jpg", save_dir)
    # reconstruct(ThreeRGBSplicer(r, r, b), optimizer, "light_rrb.jpg", save_dir)
    # reconstruct(ThreeRGBSplicer(b, g, b), optimizer, "light_bgb.jpg", save_dir)

    plt.show()