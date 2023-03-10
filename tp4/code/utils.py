import numpy as np


def read_points(file, homogenous=False):
    with open(file, 'r') as f:
        return np.array([[float(x) for x in line.split(',')] + ([1.] if homogenous else []) for line in f.readlines()])