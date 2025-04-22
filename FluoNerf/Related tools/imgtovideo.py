import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import shutil

from tqdm import tqdm
import imageio.v2 as imageio
from PIL import Image
import numpy as np

file = os.path.join(r'C:\Users\SL\Videos\light_stage/')
filelist = os.listdir(file)


def img2video():
    img_list = os.listdir(file)
    png = cv2.imread(file + '/000.png')
    size = (png.shape[1], png.shape[0])
    video = cv2.VideoWriter(file + 'output.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, size)

    for i in tqdm(img_list):
        if i.endswith('.png'):
            i = file + '/' + i
            img = cv2.imread(i)
            video.write(img)

    video.release()


def img2gif():
    P_1 = ( 'P', 'RY', 'RP', 'PB', 'BG', 'YG')
    P_2 = ('R','G','B','NCMY')
    for i in tqdm(P_2):
        img_path = r'D:\google\nerf-pytorch\logs\output_19_05_18\renderonly_path_199999/' + i #+'_video_179200\images'
        frames = []
        fps = 6
        duration = 1000 * (1/fps)
        pngFiles = os.listdir(img_path)
        # image_list = [os.path.join(file, f) for f in pngFiles]
        # exr = r'D:\google\hdr\hdr\large_corridor_4k.exr'
        # img = cv2.imread(exr,cv2.IMREAD_UNCHANGED)
        # cv2.imwrite('corridor.jpg', img)
        for image_name in tqdm(pngFiles):
            # 读取 png 图像文件
            if image_name.endswith(('.jpg', '.png')):
                frames.append(imageio.imread(os.path.join(img_path, image_name)))
        # 保存为 gif
        imageio.mimsave(img_path + '_output.gif', frames, 'GIF', duration=duration, loop=0)


def video2img():
    video_name = 'temp-08202024003502'
    video_file = file + '/' + video_name + '.avi'
    img_file = file + '/' + video_name
    if not os.path.exists(img_file):
        os.mkdir(img_file)
    video = cv2.VideoCapture(video_file)
    # flag, frame = video.read()
    frame_num = video.get(7)
    # print(len(frame))
    # if flag:

    for i in tqdm(range(int(frame_num))):
        try:
            flag, frame = video.read()
            # print(len(frame))
            img_name = img_file + '/IMG_{:06d}'.format(i) + '.jpg'
            cv2.imwrite(img_name, frame)
        except:
            pass
    video.release()


def rename():
    img_file = file + '/M/'
    file_list = os.listdir(img_file)
    file_img = [i for i in file_list if i.split(".")[-1] == "jpg" or 'png']

    for i, j in tqdm(zip(range(len(file_img)), file_img)):
        d = 0
        d += i
        new_name = img_file + '/IMG_{:06d}'.format(d) + '.jpg'
        j = img_file + j
        os.rename(j, new_name)


def img_divide(file):
    file = r'D:\google\nerf-pytorch\logs\output_19_05_18\renderonly_path_199999'
    file_img = file + '/M/'
    NR = file + '/NR'
    NG = file + '/NG'
    NB = file + "/NB"
    img_list = os.listdir(file_img)
    NR_list = os.listdir(NR)
    NG_list = os.listdir(NG)
    NB_list = os.listdir(NB)
    img_file = [i for i in img_list if i.split('.')[-1] == 'png' or 'jpg']
    # n = len(img_file) // 14
    # img = file + '/' + img_file[0]
    # img = cv2.imread(img)
    # w = img.shape[0]
    # h = img.shape[1]
    # print(img)
    for i in ('123'):
        file_C = file + '/' + i
        if not os.path.exists(file_C):
            os.mkdir(file_C)
    file_R = file + '/1/'
    file_G = file + '/2/'
    file_B = file + '/3/'
    save_num_r = [1, 2, 3, 4]
    save_num_g = [5, 6, 7, 8]
    save_num_b = [9, 10, 11, 12]
    r, g, b = [], [], []
    for k, i in enumerate(tqdm(img_file)):
        # r_name = '/IMG_{:06d}'.format(1 + i * 14) + '.jpg'
        # b_name = '/IMG_{:06d}'.format(5 + i * 14) + '.jpg'
        # g_name = '/IMG_{:06d}'.format(10 + i * 14) + '.jpg'
        # shutil.copyfile(file + r_name, file_R + r_name)
        # shutil.copyfile(file + b_name, file_B + r_name)
        # shutil.copyfile(file + g_name, file_G + r_name)
        img_i = file_img + '/' + i
        img_i = cv2.imread(img_i)
        if i in NR_list:
            cv2.imwrite(file_R + i, img_i)
        elif i in NG_list:
            cv2.imwrite(file_G + i, img_i)
        elif i in NB_list:
            cv2.imwrite(file_B + i, img_i)

        # if img_i[0, 0, 0] > 245:
        #     cv2.imwrite(file_B + i, img_i)
        # elif img_i[0, 0, 1] > 170:
        #     cv2.imwrite(file_G + i, img_i)
        # elif img_i[0, 0, 2] > 200:
        #     cv2.imwrite(file_R + i, img_i)


from PIL import Image


def resize(fin,fout):
    # fin = r"C:\Users\SL\Downloads\camera_useway\color_camera\FLIRpyspin\capture\Fixed_data\W"  # 输入图像所在路径
    # fout = r"C:\Users\SL\Downloads\camera_useway\color_camera\FLIRpyspin\capture\Fixed_data\W2"  # 输出图像的路径

    for file in tqdm(os.listdir(fin)):
        file_fullname = fin + '/' + file
        # print(file_fullname)  # 所操作图片的名称可视化
        img = Image.open(file_fullname)
        width = img.width
        height = img.height
        im_resized = img.resize((int(width/4), int(height/4)))  # resize至所需大小
        out_path = fout + '/' + file
        im_resized.save(out_path)  # 保存图像


def img_convert():
    img_file = file + '/G2/'
    file_list = os.listdir(img_file)
    file_img = [i for i in file_list if i.split(".")[-1] == "jpg" or 'png']
    for i, j in tqdm(zip(range(len(file_img)), file_img)):
        d = 0000
        d += i
        img = Image.open(img_file + j)
        new_img = img.convert("RGB", palette=Image.ADAPTIVE, colors=24)
        new_name = img_file + 'IMG_{:06d}'.format(d) + '.png'
        new_img.save(new_name)


def test_img():
    img_file = file + '/YG/'
    file_list = os.listdir(img_file)
    file_img = np.array([i for i in file_list if i.split(".")[-1] == "jpg" or 'png'])
    i_test = file_img[::8]
    # i_test = np.arange(file_img)[::8]
    file_C = img_file+'/test/'
    if not os.path.exists(file_C):
        os.mkdir(file_C)
    for i in tqdm(i_test):
        shutil.copy(img_file + i, file_C)

    return file_C


# img_divide(file)
# img2gif()
video2img()
# img2video()
# fin = test_img()
# rename()
# resize(fin, fin)
# img_convert()