import matplotlib.pyplot as plt

from stack import Stacker


if __name__ == "__main__":
    # img = plt.imread("tp2/images/Albert_Einstein.png")#[:, :, 0]
    img = plt.imread("tp2/images/orange.jpeg")/255#.sum(axis=2) / 3

    plt.figure(figsize=(16,6))

    stack = Stacker(img)
    length = stack.gaussian.shape[0]
    for i in range(length):
        ax = plt.subplot(2, length, i+1)
        ax.imshow(stack.gaussian[i], cmap="gray")
        ax.axis('off')
        
        ax = plt.subplot(2, length, length+i+1)
        ax.imshow(stack.laplacian[i][:, :, 0]+0.5, cmap="gray")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    plt.imshow(stack.laplacian.sum(axis=0))
    plt.show()
