import numpy as np
import cv2
import matplotlib.pyplot as plt


def hit_gray_img(im, L=256):
    rows, cols = im.shape[:2]
    img = im.astype(np.float)
    t = [0.0 for _ in range(L)]

    for i in range(rows):
        for j in range(cols):
            px = int(img[i, j])
            t[px] = t[px] + 1

    t = np.array([i for i in map(lambda item: item / (rows*cols), t)])  # 归一化

    plt.rc('font', family='SimHei')
    plt.xlabel('px')
    plt.subplot(3, 1, 1)
    plt.title(u'直方图')
    plt.bar(np.arange(256), t)
    # fig, ax = plt.subplots(2, 1)
    # ax[0, 0].bar(np.arange(256), t)

    # 累积直方图
    s = np.array([0.0 for _ in range(L)])
    # print(t)

    for i in range(L):
        temp = 0
        for j in t[:i]:
            temp = temp + j
        s[i] = temp
    k = ((L-1) * s + 0.5).astype(np.int)

    plt.subplot(3, 1, 2)
    plt.title(u'累积直方图')
    plt.bar(np.arange(256), s)
    # plt.show()
    # ax[0, 1].plot(np.arange(256), k)
    # fig.show()
    res = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            res[i, j] = k[im[i, j]]
    astype = res.astype(np.uint8)

    for i in range(rows):
        for j in range(cols):
            px = int(res[i, j])
            t[px] = t[px] + 1
    plt.subplot(3, 1, 3)
    plt.title(u'变换后直方图')
    plt.bar(np.arange(256), t)
    plt.show()
    return np.mat(astype)


if __name__ == '__main__':
    # im = cv2.imread(r"samples\livingroom.tif", 0)
    # L = im.max()

    # print(im.shape)
    # cv2.imshow('b', im)
    # res = hit_gray_img(im, L+1)
    # cv2.imshow('a', res)

    # cv2.waitKey(0)
    im = plt.imread(r"samples\livingroom.tif")
    plt.imshow(im, cmap='Greys_r')
    plt.show()
    L = im.max() + 1
    res = hit_gray_img(im, L)
    plt.imshow(res, cmap='Greys_r')
    plt.show()






