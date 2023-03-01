import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte
from scipy.spatial import Delaunay
from pathlib import Path

from morph import morph


def compute_avg(face_pts, img_files, saveas=None):
    imgs, pts = [], []
    for img_file, pt in zip(img_files, face_pts):
        imgs.append(img_as_float(imread(img_file)))
        h, w, _ = imgs[-1].shape

        pts.append(np.vstack([pt, [0,0], [0,h], [w,h], [w,0]]))

    pts = np.array(pts)
    avg_pts = pts.mean(axis=0)
    
    tri = Delaunay(avg_pts).simplices

    tf_imgs = []
    for i, (img, img_pts) in enumerate(zip(imgs, pts)):
        print(f"{i+1}/{len(pts)}", end='\r')
        tf_imgs.append(morph(img, img, img_pts, avg_pts, tri, 0., 1.))
    
    avg_img = img_as_ubyte(np.clip(np.array(tf_imgs).mean(axis=0), 0, 1))
    if saveas is not None:
        ax = plt.subplot(111)
        ax.triplot(avg_pts[:,0], avg_pts[:,1], tri)
        ax.set_xlim((0, avg_img.shape[1])), ax.set_ylim((avg_img.shape[0], 0))
        ax.axis('off')
        ax.set_aspect('equal', 'box')
        plt.tight_layout()
        plt.savefig(f"tp3/resultats/avg/avg_{saveas}_shape.jpg", bbox_inches='tight')
        plt.cla()
        imsave(f"tp3/resultats/avg/avg_{saveas}.jpg", avg_img)
    return avg_pts, avg_img