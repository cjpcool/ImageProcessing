import cv2
import numpy as np


def my_dct2(path):
    im = cv2.imread(path, 0)
    img1 = im.astype('float')
    rs, cs = im.shape[:2]
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
    return img_recor1


def dct_test():
    path = r"..\samples\lena_gray_512.tif"
    my_dct = my_dct2(path)

    img = cv2.imread(path, 0)
    img1 = img.astype(np.float32)

    # dct
    img_dct = cv2.dct(img1)
    img_dct_log = np.log(abs(img_dct))
    # 反变换
    img_recor2 = cv2.idct(img_dct)

    cv2.imshow("im_dct", img_dct.astype(np.uint8))
    cv2.imshow("im_dct1", img_recor2.astype(np.uint8))

    cv2.imshow('my_dct', my_dct.astype(np.uint8))
    cv2.waitKey(0)



if __name__ == '__main__':

    pass


