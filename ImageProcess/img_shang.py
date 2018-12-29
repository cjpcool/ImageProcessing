import numpy as np
import matplotlib.pyplot as plt


def get_hit(im, L=256):
    rows, cols = im.shape[:2]
    img = im.astype(np.float)
    t = [0.0 for _ in range(L)]

    for i in range(rows):
        for j in range(cols):
            px = int(img[i, j])
            t[px] = t[px] + 1

    t = np.array([i for i in map(lambda item: item / (rows*cols), t)])
    return t


def cacul_enc(t):
    x = np.array([np.log2(i) if i > 0 else 0 for i in t])
    return -1* x.dot(t.T)


if __name__ == '__main__':
    im = plt.imread(r"samples\lena_gray_512.tif")
    L = im.max() + 1
    t = get_hit(im, L)
    print(cacul_enc(t))
    im = plt.imread(r"samples\pirate.tif")
    L = im.max() + 1
    t = get_hit(im, L)
    print(cacul_enc(t))

