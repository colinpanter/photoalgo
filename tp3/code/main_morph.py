from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte

from morph import morph


if __name__ == "__main__":
    h, w = 800, 600

    name1 = "22. Panter_Colin"
    with open(f"tp3/donnees/classe/{name1}.txt", 'r') as f:
        points1 = np.array([[float(p) for p in line[1:-1].split('\t')] for line in f.readlines()] + [[0,0], [0,h], [w,h], [w,0]])
    
    name2 = "23. Perron_William"
    with open(f"tp3/donnees/classe/{name2}.txt", 'r') as f:
        points2 = np.array([[float(p) for p in line[1:-1].split('\t')] for line in f.readlines()] + [[0,0], [0,h], [w,h], [w,0]])

    img1 = img_as_float(imread(f"tp3/donnees/classe/{name1}.jpg"))
    img2 = img_as_float(imread(f"tp3/donnees/classe/{name2}.jpg"))

    N = 125
    a = 6
    sigmoid = lambda x : 1 / (1 + np.exp(-a * x))
    min_sig, max_sig = sigmoid(-0.5), sigmoid(0.5)
    dissolve_fct = lambda x : (sigmoid(x - 0.5) - min_sig) / (max_sig - min_sig)

    points = (points1 + points2) / 2
    tri = Delaunay(points)
    for n, f in enumerate(np.linspace(0, 1, N)[::-1]):
        print(f"{n+1}/{N}", end='\r')
        morphed_img = morph(img1, img2, points1, points2, tri.simplices, f, dissolve_fct(f))
        imsave(f"tp3/images/img_{n:05}.jpg", img_as_ubyte(np.clip(morphed_img, 0, 1)))
