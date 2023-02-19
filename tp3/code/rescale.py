from skimage.io import imsave, imread
from skimage.transform import resize


if __name__=="__main__":
    f = "tp3/images/Panter_Colin_original.jpg"
    img = imread("tp3/images/Panter_Colin_original.jpg")
    imsave("tp3/images/Panter_Colin.jpg", resize(img, (720, 720)))
