# 中值滤波, 边缘检测(sobel算子)
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


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
    :param im:
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



if __name__ == '__main__':
    im = plt.imread(r'samples\noise.jpg', 0)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mid_blur = mid_blur(im)
    # fig = plt.figure()
    # fig.add_subplot(221)
    # # fig.set_title('mid blur')
    # fig.imshow(mid_blur)
    # plt.show()
    im2 = plt.imread(r'samples\lena_gray_512.tif')
    edge = edge_detect(im2)
    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(im2, cmap='Greys_r')
    ax[0,0].set_title('original')
    ax[0,1].imshow(im, cmap='Greys_r')
    ax[0,1].set_title('orginal')

    ax[1,0].imshow(edge, cmap='Greys_r')
    ax[1,0].set_title('edge')
    ax[1, 1].imshow(mid_blur, cmap='Greys_r')
    ax[1, 1].set_title('mid blur')

    plt.show()




