import sqlite3
import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm.contrib import tzip


def view_data():
    # creating file path
    dbfile = './data/nerf_llff_data/flower/database.db'
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(dbfile)

    # creating cursor
    cur = con.cursor()
    print(cur.execute())
    # reading all table names
    table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    # here is you table list
    print(table_list)

    # Be sure to close the connection
    con.close()

def make_mask_data():
    P_1 = ( 'W','P', 'RY', 'RP', 'GB', 'BG', 'YG')
    P_2 = ('W','P_255', 'RY', 'RP', 'GB_255', 'BG_255', 'YG_255')
    for i, j in tzip(P_1, P_2):
        file_gt =r'E:\Download\camera_useway\color_camera\FLIRpyspin\capture\Fix_data\Fixed_data_re_6/'+ j +'/test/'
        file_CMY = r'D:\google\nerf-pytorch\logs\accv_result\Fixed_re\output_re_6\CMY\testset_150000/'+ i
        file_mask =r'D:\google\nerf-pytorch\logs\accv_result\Fixed_re\output_re_6\output_08_19_45\test_imgs_179199'
        reslut_gt = 'result_gt_b/'
        if not os.path.exists(reslut_gt): os.mkdir(reslut_gt)
        reslut_CMY = 'result_CMY_b/'
        if not os.path.exists(reslut_CMY): os.mkdir(reslut_CMY)
        img_gt_f = file_gt + '/112.png'
        img_mask_f = file_mask + '/7_fine_acc_alpha.png'
        img_CMY_f = file_CMY +'/014.png'
        img_mask = cv2.imread(img_mask_f,0)
        img_mask_w =255 - cv2.imread(img_mask_f)

        img_gt = cv2.imread(img_gt_f)
        img_CMY = cv2.imread(img_CMY_f)

        ret,mask = cv2.threshold(img_mask, 127, 1, cv2.THRESH_BINARY)
        count_pix = np.count_nonzero(mask)
        print(count_pix) # 76353
        # no_mask_gt = cv2.add(cv2.bitwise_and(img_gt,img_gt, mask=mask), img_mask_w)
        no_mask_gt = cv2.bitwise_and(img_gt, img_gt, mask=mask)
        # no_mask_CMY = cv2.add(cv2.bitwise_and(img_CMY,img_CMY, mask=mask), img_mask_w)
        no_mask_CMY = cv2.bitwise_and(img_CMY,img_CMY, mask=mask)

        cv2.imwrite(reslut_gt+'/no_mask_gt_{}.png'.format(i),no_mask_gt)
        cv2.imwrite(reslut_CMY+'/no_mask_CMY_{}.png'.format(i),no_mask_CMY)


def diff_img():
    P_1 = ( 'W','P', 'RY', 'RP', 'GB', 'BG', 'YG')
    P_2 = ('W','P_255', 'RY', 'RP', 'GB_255', 'BG_255', 'YG_255')
    for i, j in tzip(P_1, P_2):
        file_gt =r'E:\Download\camera_useway\color_camera\FLIRpyspin\capture\Fix_data\Fixed_data_re_7/'+ j +'/test/'
        file_CMY = r'D:\google\nerf-pytorch\logs\accv_result\Fixed_re\output_re_7\CMY\testset_080000/'+ i +'/'
        file_RGB = r'D:\google\nerf-pytorch\logs\accv_result\Fixed_re\output_re_7\R\R_render/'+ i +'/test/'

        alpha_path = r'D:\Shilin\project\NeRD-Neural-Reflectance-Decomposition\logs\output_01_05_04\Mask_test_imgs_179199/'
        mask_path =r'D:\Shilin\project\NeRD-Neural-Reflectance-Decomposition\data\llff\Fixed_data_re_6\masks_4_train/'

        path_CMY = './check_diff_CMY/' + i
        path_RGB = './check_diff_RGB/' + i
        if not os.path.exists(path_CMY):os.makedirs(path_CMY)
        if not os.path.exists(path_RGB): os.makedirs(path_RGB)

        #alpha_file = [n for n in os.listdir(alpha_path) if 'fine' in n]
        #b = int(alpha_file[3].split('_fine')[0])
        # alpha_file = sorted(alpha_file,key=lambda x:int(x.split('_fine')[0]))
        #alpha =  alpha_file.sort(key=lambda x:int(x.split('_fine')[0]))
        #mask_file = os.listdir(mask_path)
        GT_file = os.listdir(file_gt)
        CMY_file = os.listdir(file_CMY)
        RGB_file = [n for n in os.listdir(file_RGB) if n.endswith('.png')]

        img_a = np.zeros((1,3))
        scores = np.zeros([1,366,484,3])
        for i,alpha in enumerate(tzip(GT_file, CMY_file, RGB_file)):
            img_alpha = cv2.imread(file_gt + alpha[0])
            img_CMY = cv2.imread(file_CMY + alpha[1])
            img_RGB = cv2.imread(file_RGB + alpha[2])

            im_diff_RGB = img_RGB.astype(int) - img_alpha.astype(int)
            im_diff_CMY = img_CMY.astype(int) - img_alpha.astype(int)
            # score = cv2.absdiff(img_alpha,img_mask)
            # scores = np.concatenate((scores,[score]),axis=0)
            # diff = [np.average(np.average(score,0),0)]
            # img_a = np.concatenate((img_a,diff),axis=0)
            img_1 = cv2.imwrite(path_CMY + '/img_{:03d}.png'.format(i), im_diff_CMY)
            img_2 = cv2.imwrite(path_RGB + '/img_{:03d}.png'.format(i), im_diff_RGB)
            # print(score)
            # print(diff)
        # img_average = np.average(img_a,0)
        # diff=[i for i,n in enumerate(img_a) if n[0]>img_average[0]]
        # print(diff)
        #for i in diff:
        #    img = cv2.imwrite(path + '/img_{:03d}.png'.format(i), scores[i])
            #cv2.imshow('1',img)

        #print(img)

def specular_reflection():

    for i,j in enumerate(range(80, 100)):
        # image = input('image_path:')
        image = r'C:\Users\SL\Pictures\dataset\Flu_car_single\B\0'+ str(j) +'.png'
        pix_width = int(input('pix_width:'))
        pix_height = int(input('pix_high:'))
        img_r = cv2.imread(image)[pix_height:pix_height+3, pix_width:pix_width+3, :]
        mean_pix = np.mean(np.mean(img_r, 0), 0)
        print(mean_pix)
        img = np.ones((10, 10, 3))
        img *= mean_pix
        cv2.imwrite('pix/{:03d}.png'.format(i), img)
        draw_plot(mean_pix[0],mean_pix[1],mean_pix[2],i)


def draw_plot(r,g,b,x):
    # x = np.array(range(10,26))
    plt.xticks(range(40)[::2])
    plt.xlim(-1,21)
    plt.ylim(0,255)
    plt.scatter(x, r, color = 'red', label='R channel')
    plt.scatter(x, g, color='green', label='G channel')
    plt.scatter(x, b, color='blue', label='B channel')
    plt.xlabel('(theta,phi)')
    plt.ylabel('intensity')
    # plt.axis([])
    plt.show()


specular_reflection()

# draw_plot(110,150,20,1)
# diff_img()
# make_mask_data()