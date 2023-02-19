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

    img1 = plt.imread(f"tp3/donnees/{name1}.jpg")
    img2 = plt.imread(f"tp3/donnees/{name2}.jpg")
    
    # points = (points1 + points2) / 2
    points = points1

    tri = Delaunay(points)

    warp_frac, dissolve_frac = .9, 1.
    morphed_img = morph(img1, img2, points1, points2, tri.simplices, warp_frac, dissolve_frac)
    plt.imshow(morphed_img / 255)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # ax = plt.subplot(121)
    # ax.imshow(img1)

    # ax.triplot(points1[:,0], points1[:,1], tri.simplices)
    # ax.plot(points1[:,0], points1[:,1], 'o')

    # ax.set_xlim((0, 720))
    # ax.set_ylim((720, 0))
    # ax.axis('off')
    
    # ax = plt.subplot(122)
    # ax.imshow(img2)

    # ax.triplot(points2[:,0], points2[:,1], tri.simplices)
    # ax.plot(points2[:,0], points2[:,1], 'o')

    # ax.set_xlim((0, 720))
    # ax.set_ylim((720, 0))
    # ax.axis('off')

    # plt.tight_layout()
    # plt.show()
