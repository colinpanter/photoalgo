import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte

from utils import *
from morph import morph
from scipy.spatial import Delaunay
from random import randrange


N_COMPONENTS = 10


def create_pca():
    samples_pts = []
    for pts_file in Path(f"tp3/donnees/dlib_utrecht").glob("*.txt"):
        with open(pts_file, "r") as f:
            samples_pts.append([[float(i) for i in l[:-1].split(' ')] for l in f.readlines()])
    samples_pts = np.array(samples_pts)
    samples_features = samples_pts.reshape((samples_pts.shape[0], -1))

    pca = PCA(n_components=N_COMPONENTS, whiten=True)
    pca.fit(samples_features)

    return pca


def compare_mf(pca:PCA):
    m_pts = [read_pts_file(f) for f in get_male_files()[0]]
    f_pts = [read_pts_file(f) for f in get_female_files()[0]]

    m_features = []
    for pts in m_pts:
        m_features.append(pca.transform(pts.reshape((1, -1))))

    f_features = []
    for pts in f_pts:
        f_features.append(pca.transform(pts.reshape((1, -1))))

    m_avg_components = np.array(m_features).mean(axis=0)
    f_avg_components = np.array(f_features).mean(axis=0)

    return m_avg_components, f_avg_components


def colin_to_f(pca:PCA):
    with open("tp3/donnees/other/colin_utrecht.txt", 'r') as f:
        colin_pts = np.array([[int(i) for i in l[:-1].split(' ')] for l in f.readlines()])
    colin_features = colin_pts.reshape((1, -1))
    colin_img = img_as_float(imread("tp3/donnees/other/colin_utrecht.jpg"))

    colin_components = pca.transform(colin_features)
    m_components, f_components = compare_mf(pca)
    f_pts = pca.inverse_transform(colin_components - m_components + f_components).reshape((-1, 2))

    h, w, _ = colin_img.shape
    colin_pts = np.vstack([colin_pts, [0,0], [0,h], [w,h], [w,0]])
    f_pts = np.vstack([f_pts, [0,0], [0,h], [w,h], [w,0]])

    tri = Delaunay((colin_pts + f_pts) / 2).simplices
    
    colin_f = morph(colin_img, colin_img, colin_pts, f_pts, tri, 0., 1.)
    imsave("tp3/resultats/colin_f_v2.jpg", colin_f)


def visualize_effect(pca:PCA, component:int):
    with open("tp3/donnees/other/colin_utrecht.txt", 'r') as f:
        colin_pts = np.array([[float(i) for i in l[:-1].split(' ')] for l in f.readlines()])
    colin_img = img_as_float(imread("tp3/donnees/other/colin_utrecht.jpg"))

    colin_features = colin_pts.reshape((1, -1))
    colin_components = pca.transform(colin_features)
    delta_components = np.zeros(colin_components.shape)

    h, w, _ = colin_img.shape
    colin_pts = np.vstack([pca.inverse_transform(colin_components).reshape((-1, 2)), [0,0], [0,h], [w,h], [w,0]])
    tri = Delaunay(colin_pts).simplices

    N = 50
    for n, delta in enumerate(np.linspace(-2, 2, N)):
        print(f"Component {component}: {n+1}/{N}", end='\r')
        delta_components[..., component] = delta

        pts = pca.inverse_transform(colin_components + delta_components).reshape((-1, 2))
        pts = np.vstack([pts, [0,0], [0,h], [w,h], [w,0]])

        img = morph(colin_img, colin_img, colin_pts, pts, tri, 0., 1.)
        imsave(f"tp3/images/component_{component}/img_{n:05}.jpg", img_as_ubyte(np.clip(img, 0, 1)))


def pca_colin(pca:PCA, delta:list):
    with open("tp3/donnees/other/colin_utrecht.txt", 'r') as f:
        colin_pts = np.array([[float(i) for i in l[:-1].split(' ')] for l in f.readlines()])
    colin_img = img_as_float(imread("tp3/donnees/other/colin_utrecht.jpg"))

    colin_features = colin_pts.reshape((1, -1))
    colin_components = pca.transform(colin_features)

    print("Components of Colin\n  " + "\n  ".join([f"{i}: {x:.3}" for i, x in enumerate(colin_components[0])]))

    h, w, _ = colin_img.shape
    colin_pts = np.vstack([pca.inverse_transform(colin_components).reshape((-1, 2)), [0,0], [0,h], [w,h], [w,0]])
    tri = Delaunay(colin_pts).simplices

    delta_components = np.array(delta).reshape(colin_components.shape)

    tf_pts = pca.inverse_transform(colin_components + delta_components).reshape((-1, 2))
    tf_pts = np.vstack([tf_pts, [0,0], [0,h], [w,h], [w,0]])

    img = img_as_ubyte(np.clip(morph(colin_img, colin_img, colin_pts, tf_pts, tri, 0., 1.), 0, 1))
    imsave("tp3/resultats/pca/img.jpg", img)


if __name__ == "__main__":
    pca = create_pca()

    delta = [randrange(-3, 3) for _ in range(N_COMPONENTS)]
    delta[0], delta[1] = 0, 0
    pca_colin(pca, delta)
    print(f"Delta {delta}")
    
    # for i in range(N_COMPONENTS):
    #     Path(f"tp3/images/component_{i}").mkdir(exist_ok=True)
    #     visualize_effect(pca, i)
