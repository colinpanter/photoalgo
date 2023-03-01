from pathlib import Path

from compute_avg import compute_avg
from utils import *


if __name__ == "__main__":
    data = Path("tp3/donnees")

    print("Moyenne classe manuel")
    name = "classe_manuel"
    pts_files, imgs_files = get_all_classe_manual_files()
    pts = [read_manual_pts_file(file) for file in pts_files]
    compute_avg(pts, imgs_files, name)
    
    print("Moyenne classe dlib")
    name = "classe_dlib"
    pts_files, imgs_files = get_all_classe_dlib_files()
    pts = [read_pts_file(file) for file in pts_files]
    compute_avg(pts, imgs_files, name)
    
    print("Moyenne Utrecht")
    name = "utrecht"
    pts_files, imgs_files = get_all_utrecht_files()
    pts = [read_pts_file(file) for file in pts_files]
    compute_avg(pts, imgs_files, name)
    
    print("Moyenne avec sourire")
    name = "smile"
    pts_files, imgs_files = get_smiles_files()
    pts = [read_pts_file(file) for file in pts_files]
    compute_avg(pts, imgs_files, name)
    
    print("Moyenne sans sourire")
    name = "nosmile"
    pts_files, imgs_files = get_nosmiles_files()
    pts = [read_pts_file(file) for file in pts_files]
    compute_avg(pts, imgs_files, name)
    
    # print("Moyenne classe dlib")
    # name = "classe_dlib"
    # pts_files = list(data.joinpath("dlib_classe").glob("*.txt"))
    # imgs_files = [data.joinpath("classe", f"{file.stem}.jpg") for file in pts_files]
    # compute_avg(pts_files, imgs_files, name)
    
    # print("Moyenne Utrecht")
    # name = "utrecht"
    # pts_files = list(data.joinpath(f"dlib_{name}").glob("*.txt"))
    # imgs_files = [data.joinpath(name, f"{file.stem}.jpg") for file in pts_files]
    # compute_avg(pts_files, imgs_files, name)
    
    # print("Moyenne avec sourire")
    # name = "smile"
    # pts_files = list(data.joinpath(f"dlib_utrecht").glob("*s.txt"))
    # imgs_files = [data.joinpath("utrecht", f"{file.stem}.jpg") for file in pts_files]
    # compute_avg(pts_files, imgs_files, name)
    
    # print("Moyenne sans sourire")
    # name = "nosmile"
    # pts_files = list(data.joinpath(f"dlib_utrecht").glob("*[!s].txt"))
    # imgs_files = [data.joinpath("utrecht", f"{file.stem}.jpg") for file in pts_files]
    # compute_avg(pts_files, imgs_files, name)
