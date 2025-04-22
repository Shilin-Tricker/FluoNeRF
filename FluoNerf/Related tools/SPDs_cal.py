import math
import os
import random

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from tqdm.contrib import tzip
import pandas as pd
import xlrd
import csv


def import_excel():
    exce_data = xlrd.open_workbook(r'C:\Users\SL\Desktop\SPDs\light_spectral.xlsx')
    table = exce_data.sheets()[0]
    light_fy, j = np.array([]), np.array([])
    for i in range(0, 16, 2):
        light = np.array(table.col_values(colx=i))[1::]
        light_fy = np.append(light_fy, light, axis=0)
        j = np.append(j, i)
    # light0 = np.array(table.col_values(colx=0))[1::]
    # #light1 = np.array(table.col_values(colx=1))[1::]
    # light2 = np.array(table.col_values(colx=2))[1::]
    # #light3 = np.array(table.col_values(colx=3))[1::]
    # light4 = np.array(table.col_values(colx=4))[1::]
    # #light5 = np.array(table.col_values(colx=5))[1::]
    # light6 = np.array(table.col_values(colx=6))[1::]
    # #light7 = np.array(table.col_values(colx=7))[1::]
    # light8 = np.array(table.col_values(colx=8))[1::]
    # #light9 = np.array(table.col_values(colx=9))[1::]
    # light10 = np.array(table.col_values(colx=10))[1::]
    # #light11 = np.array(table.col_values(colx=11))[1::]
    # light12 = np.array(table.col_values(colx=12))[1::]
    # #light13 = np.array(table.col_values(colx=13))[1::]
    # light14 = np.array(table.col_values(colx=14))[1::]
    # #light15 = np.array(table.col_values(colx=15))[1::]

    exce_data_2 = pd.read_csv(r'C:\Users\SL\Desktop\SPDs\CIE_std_illum_D50.csv')
    table_2 = np.array(exce_data_2)
    std_light = table_2[::, 1] * (10 ** (-3))
    power = table.cell_value(1, 2)

    # print(light1,light2,light3,light4,light5)
    x = np.array(range(360, 781))
    std_x = np.array(range(300, 830))
    fy = light_fy.reshape([8, 421])

    # fy = np.array([light0, light1, light2, light3, light4, light5,
    #                light6, light7, light8, light9, light10, light11,
    #                light12, light13, light14, light15
    #                ])

    # the formula of weight
    fy_pinv = np.linalg.pinv(fy)
    white_light = np.array([std_light[61:482]])
    W = np.dot(white_light, fy_pinv.astype(float))
    print(W)
    W_Y = np.dot(W, fy)
    W_Y = np.squeeze(W_Y)

    white_light = np.squeeze(white_light)
    plt.figure(figsize=(15, 8))
    plt.xlim(300, 790)
    x_locator = MultipleLocator(40)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_locator)
    for k, i in enumerate(fy):
        col = (np.random.random(), np.random.random(), np.random.random())
        plt.plot(x, i, color=col, label='light_' + str(j[k]))
        # j += 1
    # plt.plot(x, light0, color='purple',label='0')
    # plt.plot(x, light1, color='gray',label='1')
    # plt.plot(x, light2, color='red',label='2')
    # plt.plot(x, light3, color='green',label='3')
    # plt.plot(x, light4, color='blue',label='4')
    plt.plot(x, W_Y, color='red', label='result')

    plt.plot(std_x, std_light, color='yellow', label='D50')
    plt.legend()
    plt.show()


def wavelength_draw():
    mean_U, std_dev_U = 390, 10
    mean_V, std_dev_V = 400, 10
    mean_B, std_dev_B = 440, 26
    mean_G, std_dev_G = 540, 30
    mean_R, std_dev_R = 620, 28

    x = np.array(range(320, 700))
    # W = 80 + 10000*np.sin(2*np.pi)
    white_light = np.random.randint(70, 80, size=380)

    U = 2000 / (std_dev_U * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_U) ** 2 / (2 * std_dev_U ** 2))
    V = 2000 / (std_dev_V * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_V) ** 2 / (2 * std_dev_V ** 2))
    B = 5000 / (std_dev_B * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_B) ** 2 / (2 * std_dev_B ** 2))
    G = 5000 / (std_dev_G * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_G) ** 2 / (2 * std_dev_G ** 2))
    R = 5000 / (std_dev_R * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_R) ** 2 / (2 * std_dev_R ** 2))

    fy = np.array([U, V, B, G, R])
    fy_pinv = np.linalg.pinv(fy)
    white_light = np.array([white_light])
    W = np.dot(white_light, fy_pinv.astype(float))
    print(W)
    # for x in range(380,700,20):
    #     U = 2000 / (std_dev_U * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_U) ** 2 / (2 * std_dev_U ** 2))
    #     V = 2000 / (std_dev_V * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_V) ** 2 / (2 * std_dev_V ** 2))
    #     B = 5000 / (std_dev_B * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_B) ** 2 / (2 * std_dev_B ** 2))
    #     G = 5000 / (std_dev_G * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_G) ** 2 / (2 * std_dev_G ** 2))
    #     R = 5000 / (std_dev_R * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_R) ** 2 / (2 * std_dev_R ** 2))
    #     W = 80 + 10*np.sin(x*np.pi)
    W_Y = np.dot(W, fy)
    W_Y = np.squeeze(W_Y)
    white_light = np.squeeze(white_light)
    plt.plot(x, U, color='purple')
    plt.plot(x, V, color='gray')
    plt.plot(x, R, color='red')
    plt.plot(x, G, color='green')
    plt.plot(x, B, color='blue')
    plt.plot(x, W_Y, color='black')
    plt.plot(x, white_light, color='yellow')
    plt.show()


def read_local():
    img_dir = r'E:\Download\SPDs\local\data_2/images/'
    folder = r'E:\Download\SPDs\local\data'
    wavelength, target_light = read_target_csv()
    light_stage = read_local_csv()

    # x = np.array(range(310, 850))
    fy = light_stage.reshape([16, 1230]).astype(float)
    target_light = target_light.reshape([2, 1230]).astype(float)
    fy_pinv = np.linalg.pinv(fy)
    # white_light = np.array([target_light])
    for n, m in enumerate(target_light):
        W = np.dot(m, fy_pinv.astype(float))
        print(W)

        W_Y = np.dot(W, fy)
        img_merge(n, W)
        plt.figure(figsize=(15, 8))
        plt.xlabel('Wavelength')
        plt.ylabel('Relative Intensity')
        # plt.xlim(300, 870)
        # x_locator = MultipleLocator(40)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(x_locator)

        for k, i in enumerate(fy):
            col = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(wavelength, i, color=col, label='light_' + str(k))
            # j += 1
        # plt.plot(x, light0, color='purple',label='0')
        # plt.plot(x, light1, color='gray',label='1')
        # plt.plot(x, light2, color='red',label='2')
        # plt.plot(x, light3, color='green',label='3')
        # plt.plot(x, light4, color='blue',label='4')
        plt.plot(wavelength, W_Y, color='red', label='result', linewidth='2')

        plt.plot(wavelength, m, color='black', label='Target', linewidth='2')
        plt.legend()
        plt.savefig(img_dir + 'result_light_'+str(n)+'.png')
        plt.show()


def read_local_csv():

    img_dir = r'E:\Download\SPDs\local\data_2/images/'
    os.makedirs(img_dir, exist_ok=True)
    dark_dir = r'E:\Download\SPDs\local\data_2/LS_unlight'

    wavelength, mean_dark = dark_csv(img_dir, dark_dir)
    all_light = np.array([])
    # n = 0
    for n in tqdm(range(16)):

        folder = r'E:\Download\SPDs\local\data_2/LS_' + str(n)

        file = [f for f in os.listdir(folder) if f.endswith('csv')]
        light_stage, total_light =np.array([]), np.array([])

        for p, q in enumerate(file):
            cvs_file_2 = open(os.path.join(folder, q), 'r', encoding='utf-8')
            exce_data_2 = csv.reader(cvs_file_2)
            light_stage = np.array([])
            for k, l in enumerate(exce_data_2):
                # print(i, j)
                if 141 <= k <= 1370:
                    light_stage = np.append(light_stage, l[6])
            light_stage = light_stage.reshape(1230, 1)
            real_light = light_stage.astype(float) - mean_dark.astype(float)
            total_light = np.append(total_light, real_light)
        total_light = total_light.reshape(p + 1, 1230, 1)
        # mean_light = np.std(total_light, ddof=1, axis=0)
        mean_light = np.mean(total_light, axis=0)
        # mean_light[615:] = 0
        # mean_light[mean_light < 80] = 0
        mean_light = np.squeeze(mean_light)
        print(mean_light)
        all_light = np.append(all_light, mean_light)

        plt.figure(figsize=(15, 8))
        plt.ylim((-100, 5000))
        plt.xlabel('Wavelength')
        plt.ylabel('Relative Intensity')

        # 拟合
        x = np.linspace(320, 850, 1230)
        fit_para = np.polyfit(x, wavelength, 2)
        #popt, pcov = curve_fit(func3, x, mean_light)
        yvals = func(x, *fit_para)

        # model = interp1d(wavelength, mean_light,kind='cubic')
        # ys = model(xs)


        plt.plot(wavelength, mean_light, color='red', label='LS_' + str(n), linewidth='2')
        plt.legend()
        # plt.savefig(img_dir + 'LS_' + str(n) + '.png')
        plt.show()


    return all_light

def func(x, a, b,c):
   return a*np.sqrt(x)*(b*np.square(x)+c)

def img_merge(n,W):
    path = r'E:\Download\light_stage\Fix_data\Fixed_data_mult_LS_5/'
    img_list = os.listdir(path)
    img_result = path + 'result/'
    os.makedirs(img_result,exist_ok=True)
    # print(img_list[12].split('_')[0])
    img_file = [i for i in img_list if i.split("_")[0] == 'light']
    img_file.sort(key=lambda x: int(x.split('_')[1]))
    img_matrix = np.array([])
    new_img = np.zeros([])
    for i in img_file:
        pic = os.path.join(path, i) + '/000.png'
        img = cv2.imread(pic)
        img_matrix = np.append(img_matrix, img)

    img_array = img_matrix.reshape([16, 8502912])  # 8,502,912
    # W[W < 0] = 0
    merge_array = np.dot(W, img_array)
    new_img = merge_array.reshape([1464, 1936, 3])
    # new_img = np.sum(img_array, 0)

    # print(len(img_array))
    # for j in range(len(img_array)):
    #     new_img = cv2.add(new_img, img_array[j] * W[j])

    img_name = img_result + '/merge_neu_dot_'+str(n)+'.png'
    cv2.imwrite(img_name, new_img)

def dark_csv(img_dir, dark_dir):
    total_dark = np.array([])
    img_dir = img_dir  # r'E:\Download\SPDs\local\data_2/images/'
    # os.makedirs(img_dir, exist_ok=True)
    dark_dir = dark_dir  # r'E:\Download\SPDs\local\data_2/LS_unlight'
    dark_csv = [f for f in os.listdir(dark_dir)]
    for c, d in tqdm(enumerate(dark_csv)):
        wavelength, dark_light = np.array([]), np.array([])
        cvs_file = open(os.path.join(dark_dir, d), 'r', encoding='utf-8')
        exce_data = csv.reader(cvs_file)
        for i, j in enumerate(exce_data):
            # print(i, j)
            if 141 <= i <= 1370:
                wavelength = np.append(wavelength, j[1])
                dark_light = np.append(dark_light, j[6])

        # data_1 = dark_data.values
        # table_1 = np.array(dark_data)
        # table_1 = table_1[6][142:1370]
        dark_light = dark_light.reshape(1230, 1)
        wavelength = wavelength.reshape(1230, 1).astype(float)
        total_dark = np.append(total_dark, dark_light)
    total_dark = total_dark.reshape(c + 1, 1230, 1).astype(float)
    mean_dark = np.mean(total_dark, axis=0)

    x = np.linspace(320, 850, 1230)
    plt.figure(figsize=(15, 8))
    plt.ylim((-100, 5000))
    plt.plot(wavelength, mean_dark, color='red', label='LS_dark', linewidth='2')
    # plt.savefig(img_dir + 'LS_dark.png')
    plt.show()

    return wavelength, mean_dark

def dark_every():
    dir = r'E:\Download\SPDs\local\data_2\LS_unlight_vex'
    folder = os.listdir(dir)
    img_dir = r'E:\Download\SPDs\local\data_2/dark_images/'
    os.makedirs(img_dir, exist_ok=True)
    dark_ex = np.array([])
    for n in tqdm(folder):
        sup_dir = os.path.join(dir, n)
        file = [f for f in os.listdir(sup_dir) if f.endswith('csv')]
        light_stage, total_light = np.array([]), np.array([])

        for p, q in enumerate(file):
            cvs_file_2 = open(os.path.join(sup_dir, q), 'r', encoding='utf-8')
            exce_data_2 = csv.reader(cvs_file_2)
            light_stage = np.array([])
            wavelength = np.array([])
            for k, l in enumerate(exce_data_2):
                # print(i, j)
                if 141 <= k <= 1370:
                    wavelength = np.append(wavelength, l[1])
                    light_stage = np.append(light_stage, l[6])

            light_stage = light_stage.reshape(1230, 1)
            wavelength = wavelength.reshape(1230, 1).astype(float)
            real_light = light_stage.astype(float)
            total_light = np.append(total_light, real_light)
        total_light = total_light.reshape(p + 1, 1230, 1)
        mean_light = np.std(total_light,ddof=1,axis= 0)
        # mean_light[615:] = 0
        # mean_light[mean_light < 80] = 0
        mean_light = np.squeeze(mean_light)
        print(mean_light)
        dark_ex = np.append(dark_ex,mean_light)

        plt.figure(figsize=(15, 8))
        plt.ylim((-100, 5000))
        plt.xlabel('Wavelength')
        plt.ylabel('Relative Intensity')

        plt.plot(wavelength, mean_light, color='red', label='LS_dark_' + str(n), linewidth='2')
        plt.legend()
        # plt.savefig(img_dir + 'LS_dark_' + str(n) + '.png')
        plt.show()

    return dark_ex

def read_target_csv():
    dir = r'E:\Download\SPDs\local\data_2'
    img_dir = r'E:\Download\SPDs\local\data_2/target_images/'
    os.makedirs(img_dir, exist_ok=True)
    folder = [f for f in os.listdir(dir) if f.endswith('ex0.15')]
    target_light = np.array([])
    dark_0 = dark_every()[0]
    for m,n in tqdm(enumerate(folder)):
        sup_dir = os.path.join(dir, n)
        file = [f for f in os.listdir(sup_dir) if f.endswith('csv')]
        light_stage, total_light = np.array([]), np.array([])

        for p, q in enumerate(file):
            cvs_file_2 = open(os.path.join(sup_dir, q), 'r', encoding='utf-8')
            exce_data_2 = csv.reader(cvs_file_2)
            light_stage = np.array([])
            wavelength = np.array([])
            for k, l in enumerate(exce_data_2):
                # print(i, j)
                if 141 <= k <= 1370:
                    wavelength = np.append(wavelength, l[1])
                    light_stage = np.append(light_stage, l[6])

            light_stage = light_stage.reshape(1230, 1)
            wavelength = wavelength.reshape(1230, 1).astype(float)
            real_light = light_stage.astype(float) - dark_0
            total_light = np.append(total_light, real_light)
        total_light = total_light.reshape(p + 1, 1230, 1)
        mean_light = np.mean(total_light, 0)
        # mean_light[615:] = 0
        mean_light[mean_light < 80] = 0
        mean_light = np.squeeze(mean_light)
        print(mean_light)
        target_light = np.append(target_light, mean_light)

        plt.figure(figsize=(15, 8))
        plt.ylim((-100, 5000))
        plt.plot(wavelength, mean_light, color='red', label='LS_dark', linewidth='2')
        # plt.savefig(img_dir + 'LS_target'+str(m)+'.png')
        plt.show()
    return wavelength, target_light

# wavelength_draw()
# import_excel()

# read_local()
read_local_csv()
# read_target_csv()

# dark_every()

# W =np.array([-3.87698059, -2.23050607, -0.16208375,  3.09081479,  1.07490275,  1.07490275,
#   3.39183582,  1.91602881,  0.67629227,  1.1458617,   0.59707243 , 0.02102463,
#  -0.31511752, -0.70475084, -0.25796417, -2.39411662])
#
#
# img_merge(0,W)
