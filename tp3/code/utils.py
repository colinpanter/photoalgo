import numpy as np
from pathlib import Path


DATA = Path("tp3/donnees")

def read_pts_file(file:Path):
    with open(file, 'r') as f:
        pts = np.array([[float(i) for i in l[:-1].split(' ')] for l in f.readlines()])
    return pts

def read_manual_pts_file(file:Path):
    with open(file, 'r') as f:
        pts = np.array([[float(p) for p in line[1:-1].split('\t')] for line in f.readlines()])
    return pts

def get_all_classe_manual_files():
    pts_files = list(DATA.joinpath(f"classe").glob("*. *.txt"))
    imgs_files = [DATA.joinpath("classe", f"{file.stem}.jpg") for file in pts_files]
    return pts_files, imgs_files

def get_all_utrecht_files():
    pts_files = list(DATA.joinpath(f"dlib_utrecht").glob("*.txt"))
    imgs_files = [DATA.joinpath("utrecht", f"{file.stem}.jpg") for file in pts_files]
    return pts_files, imgs_files

def get_all_classe_dlib_files():
    pts_files = list(DATA.joinpath(f"dlib_classe").glob("*.txt"))
    imgs_files = [DATA.joinpath("classe", f"{file.stem}.jpg") for file in pts_files]
    return pts_files, imgs_files

def get_smiles_files():
    pts_files = list(DATA.joinpath(f"dlib_utrecht").glob("*s.txt"))
    imgs_files = [DATA.joinpath("utrecht", f"{file.stem}.jpg") for file in pts_files]
    return pts_files, imgs_files

def get_nosmiles_files():
    pts_files = list(DATA.joinpath(f"dlib_utrecht").glob("*[!s].txt"))
    imgs_files = [DATA.joinpath("utrecht", f"{file.stem}.jpg") for file in pts_files]
    return pts_files, imgs_files

def get_female_files():
    pts_files = list(DATA.joinpath(f"dlib_utrecht").glob("f*.txt"))
    imgs_files = [DATA.joinpath("utrecht", f"{file.stem}.jpg") for file in pts_files]
    return pts_files, imgs_files

def get_male_files():
    pts_files = list(DATA.joinpath(f"dlib_utrecht").glob("m*.txt"))
    imgs_files = [DATA.joinpath("utrecht", f"{file.stem}.jpg") for file in pts_files]
    return pts_files, imgs_files


if __name__ == "__main__":
    print([f for f in get_male_files()[0]])
