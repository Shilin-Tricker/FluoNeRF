import math
import os

import cv2
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip

# path = r'E:\ShiLin\Project\nerf-pytorch\data\llff\temp-NR\images_2\\IMG_000255.jpg'
# png = cv2.imread(path)
# print(png)

file = r'D:\google\nerf-pytorch\Fixed_data_LS\reslut_2\testset_1/R/'
file_2 = r'D:\google\nerf-pytorch\Fixed_data_LS\reslut_2\testset_1/G/'
file_3 = r'D:\google\nerf-pytorch\Fixed_data_LS\reslut_2\testset_1/B/'
file_4 = r'D:\google\nerf-pytorch\Fixed_data_LS\reslut_2\testset_1/U/'
file_5 = r'D:\google\nerf-pytorch\Fixed_data_LS\reslut_2\testset_1/V/'


def rename():
    file_name = file
    file_list = os.listdir(file_name)
    img_file = [i for i in file_list if i.split(".")[-1] == 'png']
    for i, j in tqdm(zip(range(len(img_file)), img_file)):
        new_name = file_name + 'IMG_{:06d}'.format(i) + '.jpg'
        j = file_name + j
        os.rename(j, new_name)


def IMG_Enhence():
    img_list = os.listdir(file)
    img_file = [i for i in img_list if i.split(".")[-1] == 'png' or 'jpg']
    for k in tqdm(img_file):
        k = file + k
        img = cv2.imread(k)  # BGR
        width = img.shape[0]
        heigh = img.shape[1]
        # w_img = np.zeros((1024, 1280))
        # img[:, :, 2] = 0
        img = img * 10
        # for i in range(width):
        #     for j in range(heigh):
        #         for a in range(3):
        #             img[i][j][a] = min(255, img[i][j][a] / 2)  # B up 1.3
                # if img[i][j][0] < 16:  # 0:B ; 1:G ; 2:R
                #    img[i][j] = 0
        #         # if img[i][j][1] == 0:
        #         #     img[i][j] += 25
        cv2.imwrite(k, img)


def Img_merge():
    global PSNR, SSIM
    img_r = os.listdir(file)
    img_g = os.listdir(file_2)
    img_b = os.listdir(file_3)
    img_w = os.listdir(file_4)
    img_v = os.listdir(file_5)
    img_file_r = [i for i in img_r if i.split(".")[-1] == 'png' or 'jpg']
    img_file_g = [i for i in img_g if i.split(".")[-1] == 'png' or 'jpg']
    img_file_b = [i for i in img_b if i.split(".")[-1] == 'png' or 'jpg']
    img_file_w = [i for i in img_w if i.split(".")[-1] == 'jpg' or 'png']
    img_file_v = [i for i in img_v if i.split(".")[-1] == 'jpg' or 'png']
    phi = 180.
    for k, a, b, w, v in tzip(img_file_r, img_file_g, img_file_b, img_file_w, img_file_v):
        # i_r = round(np.random.random(), 2)
        # i_g = round(np.random.random(), 2)
        # i_b = round(np.sqrt(abs(1 - i_r **2 - i_g ** 2)), 1)
        # random color
        # i_r = np.random.randint(2)
        # i_g = np.random.randint(2)
        # i_b = np.random.randint(2)

        # i_b = round(abs(1 - i_r ** 2 - i_g ** 2), 1)
        k = file + k
        # n_file = file + 'N_M' + k
        a = file_2 + a
        b = file_3 + b
        w = file_4 + w
        v = file_5 + v
        k_r = cv2.imread(k)  # BGR
        # print(k_r.dtype)
        a_g = cv2.imread(a)
        b_b = cv2.imread(b)
        # print(b_b)
        w_u = cv2.imread(w)
        w_v = cv2.imread(v)
        width = k_r.shape[0]
        heigh = k_r.shape[1]
        # w_w[:, :, 0] = w_w[:, :, 0] * 0.5
        # w_w[:, :, 1] = 0

        #cv2.imwrite(w, w_w)

        k_w = cv2.add(w_u, w_v, dtype=cv2.CV_8UC3)
        # k_w = cv2.add(k_w, k_r)
        # w_k = cv2.add(a_g, w_w)
        # color editing
        # k_w = cv2.add(k_r, a_g).astype(np.uint8)
        # k_w = cv2.add(k_w, (i_b * b_b).astype(np.uint8))
        # k_w[:, :, 0] = k_w[:, :, 0] * 64
        # k_w[:, :, 1] = k_w[:, :, 1] * 128
        # k_w[:, :, 2] = k_w[:, :, 2] * 196
        # k_w = (k_w * 255).astype(np.uint8)

        cv2.imwrite(k, k_w)
        # cv2.imwrite(w, w_k)
        # phi -= 5.

        # PSNR = psnr(k_w, w_w)
        # SSIM = ssim(k_w, w_w)
        # print("PSNR:%.3f" % PSNR)
        # print('SSIM:%.3f' % SSIM)


def Img_single_merge():
    file_path = r'D:\google\nerf-pytorch\logs\output_20_02_Plant_RGB\testset_050000\R'
    img1 = r'D:\google\nerf-pytorch\logs\output_20_02_Plant_RGB\testset_050000\R/003.png'
    img2 = r'D:\google\nerf-pytorch\logs\output_20_02_Plant_RGB\testset_050000\G/003.png'
    img3 = r'D:\google\nerf-pytorch\logs\output_20_02_Plant_RGB\testset_050000\B/003.png'
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img3 = cv2.imread(img3)
    w = [1.34852547, 1.48659517, 1.24242424]
    # k_w = cv2.add(img1 * w[2], img2 * w[1])
    # k_w = cv2.add(k_w, img3*w[0])
    k_w = cv2.add(img1, img2)
    k_w = cv2.add(k_w, img3)
    cv2.imwrite(file_path + '/merge.png', k_w)


# def ssim(y_true, y_pred):
#     u_true = np.mean(y_true)
#     u_pred = np.mean(y_pred)
#     var_true = np.var(y_true)
#     var_pred = np.var(y_pred)
#     std_true = np.sqrt(var_true)
#
#     std_pred = np.sqrt(var_pred)
#     R = 255
#     c1 = np.square(0.01 * R)
#     c2 = np.square(0.03 * R)
#     ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
#     denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
#     return ssim / denom

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def front_psnr(img1,img2,count_pix):
    mse = np.sum((img1 - img2) ** 2)/(count_pix)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def img_make():
    # image = cv2.imread('G_250.png')
    # for i in range(256):
    img = np.ones((2048, 1024, 3))
    file_name = './light_map'
    if not os.path.exists(file_name): os.mkdir(file_name)
    R= cv2.imread(file_name+'/RP.png')
    G = cv2.imread(file_name + '/RY.png')
    B = cv2.imread(file_name + '/GB.png')
    W = cv2.imread(file_name + '/BG.png')
    YG = cv2.imread(file_name + '/YG.png')
    width = 2560
    high = 1440
    w_step = 640
    h_step = 360
    n = 0
    # img = np.zeros((1440, 2560, 3))
    # for j in tqdm(range(0, high, h_step)):
    #     for i in range(0, width, w_step):
    #         img = np.zeros((1440, 2560, 3))
    #         cv2.rectangle(img, (i, j), (i + w_step, j + h_step), (255, 0, 0), -1)
    #
    #         cv2.imwrite(file_name + '/{:03d}.png'.format(n), img)
    #         n += 1

    #imge = img * [0, 255, 128]  # b,g,r
    imge = cv2.hconcat([R,G,B,W,YG])
    cv2.imshow('img', imge)
    cv2.imwrite('light_map/RPYGB.png', imge)
    # cv2.imwrite('light_data/{:03d}.png'.format(i), imge)


def evaluation():

    img_1 = r'E:\Download\light_stage\Fix_data\Fixed_data_mult_LS_5\Fluolights_2/000.png'
    img_3 = r'D:\google\nerf-pytorch\logs\accv_result\Fixed_re\output_re_6\result_CMY_b\no_mask_CMY_P.png'
    # img2 = r'C:\Users\SL\Pictures\dataset\Flu_tennis_single\B2\096.png'
    img_2 = r'E:\Download\light_stage\Fix_data\Fixed_data_mult_LS_5\result\merge_neu_dot_1.png'
    img_1 = cv2.imread(img_1)
    img_2 = cv2.imread(img_2)
    img_3 = cv2.imread(img_3)

    #PSNR_1 = front_psnr(img_1, img_2, 76353)
    PSNR = psnr(img_1, img_2)
    SSIM_1 = ssim(img_1, img_2)
    #PSNR_2 = front_psnr(img_1, img_3, 76353)
    #SSIM_2 = ssim(img_1, img_3)

    # print("PSNR_nerd:%.2f" % PSNR)
    # print('SSIM_nerd:%.3f' % SSIM_1)
    print("PSNR:%.2f" % PSNR)
    print('SSIM:%.3f' % SSIM_1)


def mean_eval():
    P_1 = ('1','2','3','6','12')#('R','G','B','C','M','Y','W','P','RY','RP','GB','BG','YG')
    P_2 = ('LS_1','LS_2','B','G','R')#('R','G','B','C','M','Y','W','P_255','RY','RP','GB_255','BG_255','YG_255')
    for i,j in tzip(P_1,P_2):
        file_path_1 = r'D:\google\nerf-pytorch\Fixed_data_LS\result\testset_600000/' + j +'/'#test/'
        file_path_2 = r'E:\Download\light_stage\Fix_data\Fixed_data_LS\quarter_light_' + i +'/test/'
        file_1 = os.listdir(file_path_1)
        file_2 = os.listdir(file_path_2)
        img_file_1 = [i for i in file_1 if i.split(".")[-1] == 'png' or 'jpg']
        img_file_2 = [i for i in file_2 if i.split(".")[-1] == 'png' or 'jpg']

        arr_PSNR = []
        arr_SSIM = []
        for a, b in tzip(img_file_1, img_file_2):
            a = file_path_1 + a
            b = file_path_2 + b
            img1 = cv2.imread(a)  # BGR

            img2 = cv2.imread(b)
            PSNR = psnr(img1, img2)
            SSIM = ssim(img1, img2)
            arr_PSNR.append(PSNR)
            arr_SSIM.append(SSIM)

        mean_PSNR = np.mean(arr_PSNR)
        mean_SSIM = np.mean(arr_SSIM)
        print(i)
        print("PSNR:%.2f" % mean_PSNR)
        print('SSIM:%.3f' % mean_SSIM)


def Resize():
    from PIL import Image
    img_file = r'E:\Download\light_stage\Fix_data\Fixed_data_mult_LS\quarter_light_6_12/'
    img = Image.open(img_file + '024.png')
    width, height = img.size
    img.thumbnail((width / 4, height / 4))
    img.save(img_file + 'resize_O.png', 'png')



#Resize()
# rename()

evaluation()
# mean_eval()
# img_make()

# Img_merge()


# Img_single_merge()
# IMG_Enhence()


# minify()
