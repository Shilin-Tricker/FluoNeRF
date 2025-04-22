import math
import os
import random

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm.contrib import tzip

file = r'D:\google\nerf-pytorch\Fixed_data_LS\result\testset_060000\R\/'
file_2 = r'D:\google\nerf-pytorch\Fixed_data_LS\result\testset_060000\G\/'
file_3 = r'D:\google\nerf-pytorch\Fixed_data_LS\result\testset_060000\B\/'
file_4 = r'D:\google\nerf-pytorch\Fixed_data_LS\result\testset_060000\B\/'

img_r_path = r'E:\Download\light_stage\Fix_data\Fixed_data_LS_16\quarter_light_12\108.png'
img_g_path = r'E:\Download\light_stage\Fix_data\Fixed_data_LS_16\quarter_light_6\108.png'
img_b_path = r'E:\Download\light_stage\Fix_data\Fixed_data_LS_16\quarter_light_3\108.png'
img_u_path = r'E:\Download\light_stage\Fix_data\Fixed_data_LS_16\quarter_light_0\108.png'
img_v_path = r'E:\Download\light_stage\Fix_data\Fixed_data_LS_16\quarter_light_1\108.png'
img_y_path = r'E:\Download\light_stage\Fix_data\Fixed_data_LS_16\quarter_light_9\108.png'


def W_calculate():
    pic = cv2.imread(r'C:\Users\SL\Downloads\camera_useway\color_camera\FLIRpyspin\capture\Fixed_data_5\W\001.png')
    img_r = cv2.imread(img_r_path)[1430:1450, 1870:1890, :]
    img_g = cv2.imread(img_g_path)[1430:1450, 1870:1890, :]
    img_b = cv2.imread(img_b_path)[1430:1450, 1870:1890, :]
    img_u = cv2.imread(img_u_path)[1430:1450, 1870:1890, :]
    img_v = cv2.imread(img_v_path)[1430:1450, 1870:1890, :]
    img_y = cv2.imread(img_y_path)[1430:1450, 1870:1890, :]
    r = np.mean(np.mean(img_r, 0), 0)
    g = np.mean(np.mean(img_g, 0), 0)
    b = np.mean(np.mean(img_b, 0), 0)
    u = np.mean(np.mean(img_u, 0), 0)
    v = np.mean(np.mean(img_v, 0), 0)
    y = np.mean(np.mean(img_y, 0), 0)
    print('img_y:', y)
    print('img_r:', r)
    print('img_g:', g)
    print('img_b:', b)
    img_0 = np.array([0, 0, 0])
    fy = np.array([b, g, r, u, v])
    # fy_pinv = np.linalg.pinv(img_fy)
    W1 = np.dot(y, np.linalg.pinv(fy)).astype(float)
    W2 = np.dot(y, y).astype(float)
    print('W:', W1)
    # print('W2:', W2)
    # W[2] =0
    # print(W)
    # Img_merge(W1)


def mean_calculate():
    W = []
    for i in range(10):
        img_r = cv2.imread(img_r_path)[420 + i, 620 + i, :]
        img_g = cv2.imread(img_g_path)[420 + i, 620 + i, :]
        img_b = cv2.imread(img_b_path)[420 + i, 620 + i, :]
        img_y = cv2.imread(img_y_path)[420 + i, 620 + i, :]
        # y = np.mean(img_y)
        print('img_y:', img_y)
        img_0 = np.array([0, 0, 0])
        img_fy = np.array([img_b, img_g, img_r])
        # fy_pinv = np.linalg.pinv(img_fy)
        W1 = np.dot(img_y, np.linalg.inv(img_fy)).astype(float)
        W2 = np.dot(img_y, img_fy).astype(float)
        W.append(W1)

    mean_W1 = np.mean(W[0])
    # mean_W2 = np.mean(W2)
    print('W1:', mean_W1)
    # print('W2:', mean_W2)


def Img_merge(W):
    global PSNR, SSIM
    img_r = os.listdir(file)
    img_g = os.listdir(file_2)
    img_b = os.listdir(file_3)
    img_w = os.listdir(file_4)
    img_file_r = [i for i in img_r if i.split(".")[-1] == 'png' or 'jpg']
    img_file_g = [i for i in img_g if i.split(".")[-1] == 'png' or 'jpg']
    img_file_b = [i for i in img_b if i.split(".")[-1] == 'png' or 'jpg']
    img_file_w = [i for i in img_w if i.split(".")[-1] == 'jpg' or 'png']
    # phi = 180.

    for i, j in enumerate(tzip(img_file_r, img_file_g, img_file_b, img_file_w)):

        k = file + j[0]
        n_file = file + '/Merge_{:05d}.jpg'.format(i)
        a = file_2 + j[1]
        b = file_3 + j[2]
        w = file_4 + j[3]
        w_file = file_4 + '/Merge_{:05d}.jpg'.format(i)

        k_r = cv2.imread(k)  # BGR
        # print(k_r.dtype)
        a_g = cv2.imread(a)
        b_b = cv2.imread(b)
        # print(b_b)
        w_w = cv2.imread(w)
        width = k_r.shape[0]
        height = k_r.shape[1]
        for i in range(len(W)):
            if W[i] < 0:
                w_w[:, :, i] = 0
            else:
                W_G = w_w[:, :, i].astype('float32') / 255 * W[i]
                w_w[:, :, i] = (W_G * 255).astype(np.uint8)
                # W_B =w_w[:, :, 0].astype('float32') / 255 * W[2]
                # w_w[:, :, 0] = (W_B*255).astype(np.uint8)

        # cv2.imwrite(w_file, w_w)

        k_w = cv2.add(k_r * W[2], a_g * W[1])
        # k_w = cv2.add(k_r * W[1], b_b*W[0])
        k_w = cv2.add(k_w, b_b * W[0])

        # color editing
        # k_w = cv2.add(k_r, a_g).astype(np.uint8)
        # k_w = cv2.add(k_w, (i_b * b_b).astype(np.uint8))
        # k_w[:, :, 0] = k_w[:, :, 0] * 64
        # k_w[:, :, 1] = k_w[:, :, 1] * 128
        # k_w[:, :, 2] = k_w[:, :, 2] * 196
        # k_w = (k_w * 255).astype(np.uint8)

        cv2.imwrite(n_file, k_w)


def average_pix():
    img_r = os.listdir(file)
    img_g = os.listdir(file_2)
    img_b = os.listdir(file_3)
    img_w = os.listdir(file_4)
    img_file_r = [i for i in img_r if i.split(".")[-1] == 'png' or 'jpg']
    img_file_g = [i for i in img_g if i.split(".")[-1] == 'png' or 'jpg']
    img_file_b = [i for i in img_b if i.split(".")[-1] == 'png' or 'jpg']
    img_file_w = [i for i in img_w if i.split(".")[-1] == 'jpg' or 'png']
    pixs_r = np.array([])
    pixs_g = np.array([])
    pixs_b = np.array([])
    for i, j in enumerate(tzip(img_file_r, img_file_g, img_file_b, img_file_w)):
        k = file + j[0]
        a = file_2 + j[1]
        c = file_3 + j[2]
        w = file_4 + j[3]
        img_r = cv2.imread(k)[400:425, 700:725, :]
        img_g = cv2.imread(a)[400:425, 700:725, :]
        img_b = cv2.imread(c)[400:425, 700:725, :]
        img_y = cv2.imread(w)[400:425, 700:725, :]
        r = np.mean(np.mean(img_r, 0), 0)
        g = np.mean(np.mean(img_g, 0), 0)
        b = np.mean(np.mean(img_b, 0), 0)
        y = np.mean(np.mean(img_y, 0), 0)
        # pix_r = np.array([[10 + 2*i, r[2]]])
        # pix_g = np.array([[10 + 2*i, r[1]]])
        # pix_b = np.array([[10 + 2*i, r[0]]])
        # pix_w = [10 + 2 * i, r]

        # print('img_40:', y)
        # print('img_10:', r)
        # print('img_20:', g)
        # print('img_30:', b)
        pixs_r = np.append(pixs_r, r[2])
        pixs_g = np.append(pixs_g, g[1])
        pixs_b = np.append(pixs_b, b[0])
    draw_plot(pixs_r, pixs_g, pixs_b)


# mean_calculate()
# W_calculate()


def draw_plot(r, g, b):
    x = np.array(range(10, 26, 1))
    plt.plot(x, r, color='red')
    plt.plot(x, g, color='green')
    plt.plot(x, b, color='blue')
    # plt.axis([])
    plt.show()


def img_pix():
    img_r = cv2.imread(img_r_path)[400:425, 700:725, :]
    img_g = cv2.imread(img_g_path)[400:425, 700:725, :]
    img_b = cv2.imread(img_b_path)[400:425, 700:725, :]
    img_y = cv2.imread(img_y_path)[400:425, 700:725, :]
    # img_y = cv2.imread(img_y_path)
    # img_y = cv2.cvtColor(img_y,cv2.COLOR_BGR2RGB)
    # cv2.imwrite(file_4+'t_w_11.png',img_y)

    r = np.mean(np.mean(img_r, 0), 0)
    g = np.mean(np.mean(img_g, 0), 0)
    b = np.mean(np.mean(img_b, 0), 0)
    y = np.mean(np.mean(img_y, 0), 0)
    # pix_r = np.array([[10 + 2*i, r[2]]])
    # pix_g = np.array([[10 + 2*i, r[1]]])
    # pix_b = np.array([[10 + 2*i, r[0]]])
    # pix_w = [10 + 2 * i, r]

    print('img_w:', y)
    print('img_r:', r)
    print('img_g:', g)
    print('img_b:', b)


def wavelength_cal():
    mean_U, std_dev_U = 390, 10
    mean_V, std_dev_V = 400, 10
    mean_B, std_dev_B = 440, 26
    mean_G, std_dev_G = 540, 30
    mean_R, std_dev_R = 620, 28

    x = np.array(range(320, 700))
    white_light = np.random.randint(70, 80, size=380)

    U = 2000 / (std_dev_U * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_U) ** 2 / (2 * std_dev_U ** 2))
    V = 2000 / (std_dev_V * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_V) ** 2 / (2 * std_dev_V ** 2))
    B = 5000 / (std_dev_B * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_B) ** 2 / (2 * std_dev_B ** 2))
    G = 5000 / (std_dev_G * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_G) ** 2 / (2 * std_dev_G ** 2))
    R = 5000 / (std_dev_R * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_R) ** 2 / (2 * std_dev_R ** 2))

    plt.plot(x, U, color='purple')
    plt.plot(x, V, color='gray')
    plt.plot(x, R, color='red')
    plt.plot(x, R, color='red')
    plt.plot(x, G, color='green')
    plt.plot(x, B, color='blue')
    plt.plot(x, white_light, color='yellow')
    plt.show()


# average_pix()
W = [0.299183, 0.00434947, 0.03797259]
# img_pix()

wavelength_cal()
