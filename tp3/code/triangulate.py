from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np

from morph import morph


if __name__ == "__main__":
    name1 = "22. Panter_Colin"
    with open(f"tp3/donnees/{name1}.txt", 'r') as f:
        points1 = np.array([[float(p) for p in line[1:-1].split('\t')] for line in f.readlines()] + [[0,0], [0,720], [720,720], [720,0]])
    
    name2 = "23. Perron_William"
    with open(f"tp3/donnees/{name2}.txt", 'r') as f:
        points2 = np.array([[float(p) for p in line[1:-1].split('\t')] for line in f.readlines()] + [[0,0], [0,720], [720,720], [720,0]])

    img1 = plt.imread(f"tp3/donnees/{name1}.jpg") / 255
    img2 = plt.imread(f"tp3/donnees/{name2}.jpg") / 255
    
    # points = (points1 + points2) / 2
    points = points1

    tri = Delaunay(points)

    warp_frac, dissolve_frac = .5, .5
    N = 125
    for n, i in enumerate(np.linspace(0, 1, N)[::-1]):
        print(f"{n+1}/{N}", end='\r')
        morphed_img = morph(img1, img2, points1, points2, tri.simplices, i, i)
        plt.imshow(np.clip(morphed_img, 0, 1))
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"tp3/images/img_{n:05}.jpg")
        plt.cla()
