import numpy as np
from pathlib import Path
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte
from scipy.spatial import Delaunay

from compute_avg import compute_avg
from morph import morph
from utils import *


if __name__ == "__main__":
    data = Path("tp3/donnees")
    
    print("Average male")
    name = "male"
    pts_files, imgs_files = get_male_files()
    pts = [read_pts_file(file) for file in pts_files]
    m_pts, m_img = compute_avg(pts, imgs_files, f"{name}")
    
    print("Average female")
    name = "female"
    pts_files, imgs_files = get_female_files()
    pts = [read_pts_file(file) for file in pts_files]
    f_pts, f_img = compute_avg(pts, imgs_files, f"{name}")

    colin_img = img_as_float(imread(data.joinpath("other", "colin_utrecht.jpg")))
    h, w, _ = colin_img.shape
    with open(data.joinpath("other", "colin_utrecht.txt"), "r") as f:
        colin_pts = [[int(i) for i in l[:-1].split(' ')] for l in f.readlines()]
        colin_pts += [[0,0], [0,h], [w,h], [w,0]]
        colin_pts = np.array(colin_pts)
    
    tri = Delaunay((colin_pts + m_pts) / 2).simplices
    colin_m = morph(colin_img, img_as_float(m_img), colin_pts, m_pts, tri, 0., 1.)
    imsave("tp3/resultats/colin_m.jpg", img_as_ubyte(np.clip(colin_m, 0, 1)))

    colin_m_2 = morph(colin_img, img_as_float(m_img), colin_pts, m_pts, tri, 0.5, 0.5)
    imsave("tp3/resultats/colin_m_2.jpg", img_as_ubyte(np.clip(colin_m_2, 0, 1)))
    
    tri = Delaunay((colin_pts + f_pts) / 2).simplices
    colin_f = morph(colin_img, img_as_float(f_img), colin_pts, f_pts, tri, 0., 1.)
    imsave("tp3/resultats/colin_f.jpg", img_as_ubyte(np.clip(colin_f, 0, 1)))
    
    colin_f_2 = morph(colin_img, img_as_float(f_img), colin_pts, f_pts, tri, 0.5, 0.5)
    imsave("tp3/resultats/colin_f_2.jpg", img_as_ubyte(np.clip(colin_f_2, 0, 1)))
