import cv2
import numpy as np
import matplotlib.pyplot as plt

def arnold(im):
    x, y = im.shape[:2]
    res = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
            res[i, j] = im[(i+j) % x, (i+2*j) % y]
    return res


def wei_mian_tu(im):
    rs, cs = im.shape[:2]
    ans = tuple([np.zeros((rs, cs)) for i in range(8)])

    for r in range(rs):
        for c in range(cs):
            for i in range(7, -1, -1):
                if im[r, c] > 2**i:
                    ans[i][r, c] = np.uint8(255)
                    break
    return ans


def arnoldtest():
    path = r"samples\lena_gray_512.tif"
    im = cv2.imread(path)
    img = arnold(im)
    cv2.imshow('aa', img.astype(np.uint8))
    cv2.waitKey(0)


if __name__ == '__main__':
    path = r"samples\lena_gray_512.tif"
    im = plt.imread(path)
    # print(dir(im))
    fig = plt.figure()
    # print(im.cmap)
    ax = fig.add_subplot(1, 2, 1)
    ax.tick_params(bottom="off", top="off", left="off", right="off")
    ax.imshow(im, cmap=plt.cm.gray)

    # plt.show()
    # im = cv2.imread(path, 0)
    res = arnold(im)
    ax.bar()
    ax = fig.add_subplot(1, 2, 2)
    ax.tick_params(bottom="off", top="off", left="off", right="off")
    ax.imshow(res, cmap=plt.cm.gray)
    plt.show()