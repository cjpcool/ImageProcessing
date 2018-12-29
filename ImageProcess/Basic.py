import cv2
import numpy as np
import math

def hit_gray_img(im, L=256):
    """
    均衡化处理
    :param im:
    :param L: 图片长宽像素数量
    :return:
    """
    rows, cols = im.shape[:2]
    img = im.astype(np.float)
    t = [0.0 for _ in range(L)]

    for i in range(rows):
        for j in range(cols):
            px = int(img[i, j])
            t[px] = t[px] + 1

    t = np.array([i for i in map(lambda item: item / (rows*cols), t)])  # 归一化

    # 累积直方图
    s = np.array([0.0 for _ in range(L)])
    # print(t)
    for i in range(L):
        temp = 0
        for j in t[:i]:
            temp = temp + j
        s[i] = temp
    k = ((L-1) * s + 0.5).astype(np.int)
    res = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            res[i, j] = k[im[i, j]]
    astype = res.astype(np.uint8)
    return np.mat(astype)

def arnold(im):
    """
    arnold变换
    :param im:
    :return:
    """
    x, y, z = im.shape
    res = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
                res[i, j] = im[(i+j) % x, (i+2*j) % y]
    return res.astype(np.uint8)


def my_dct2(im):
    rs, cs = im.shape[:2]
    img1 = im.astype('float')
    N = cs
    C_tmp = np.zeros((rs, cs))

    C_tmp[0, :] = 1 * np.sqrt(1/N)
    for i in range(1, rs):
        for j in range(cs):
            C_tmp[i, j] = np.sqrt(2/N) * np.cos(np.pi * i * (2 * j + 1) / (2*N))

    dst = np.dot(C_tmp, img1)
    dst = np.dot(dst, np.transpose(C_tmp))

    dst1 = np.log(abs(dst))

    img_recor = np.dot(np.transpose(C_tmp), dst)
    img_recor1 = np.dot(img_recor, C_tmp)
    return img_recor1.astype(np.uint8)


def mid_blur(im, kernel=np.ones((3,3))):
    """
    计算中值滤波
    :param im:
    :param kernel:
    :return:
    """
    im_r, im_c = im.shape[:2]
    ans = np.array(im)
    k_r, k_c = kernel.shape
    off = int((k_r-1) / 2)  # 模板与中心的偏移量
    for i in range(off, im_r - off):
        for j in range(off, im_c - off):
            tmp = im[i-off:i+off+1, j-off:j+off+1] * kernel
            median = np.median(tmp)
            ans[i, j] = median

    return ans


def edge_detect(im, kernel=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])):
    """
    sobel算子
    :param im: np.array
    :param kernel: 默认sobel算子
    :return:
    """
    im_r, im_c = im.shape[:2]
    ans = np.zeros((im_r, im_c))
    k_r, k_c = kernel.shape
    off = int((k_r - 1) / 2)  # 模板与中心的偏移量
    for i in range(off, im_r - off):
        for j in range(off, im_c - off):
            part = im[i - off:i + off+1, j - off:j + off+1]
            tmp_x = part * kernel
            sum_x = tmp_x.sum()
            tmp_y = part * kernel.T
            sum_y = tmp_y.sum()
            ans[i, j] = math.sqrt(sum_x**2 + sum_y**2)

    return ans
