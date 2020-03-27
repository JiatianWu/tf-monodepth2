import os
import pdb
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import tensorflow as tf

from tools import *
from bilateral_filter import bilateral_filter

def debug_convert():
    img_path = '/home/jiatian/dataset/tum/sequence_01/000001.jpg'
    img = tf.read_file(img_path)
    img_gray = tf.image.decode_jpeg(img)
    img_v1 = tf.image.decode_jpeg(img, channels=3)
    img_v2 = tf.image.grayscale_to_rgb(img_gray)

    with tf.Session() as sess:
        np_img_v1 = sess.run(img_v1)
        np_img_v2 = sess.run(img_v2)

def read_imu_data():
    data_path = '/home/jiatian/dataset/office_kitchen_0001a/a-1315403270.593277-3849079182.dump'
    # with open(data_path, 'rb') as f:
    #     data = pickle.load(f)
    f = open(data_path, 'r')
    pdb.set_trace()

def resave_imu_data():
    dataset = '/freezer/nyudepthV2_raw'
    seqs = os.listdir(dataset)
    for seq in seqs:
        seq_dir = dataset + '/' + seq
        for data in os.listdir(seq_dir):
            if data[0] == 'a':
                imu_data_path = seq_dir + '/' + data
                resave_imu_data_path = seq_dir + '/' + data[:-4] + '.txt'
                call_resave_imu(imu_data_path, resave_imu_data_path)

def call_resave_imu(orig_path, resave_path):
    command = './resave_imu ' + orig_path + ' ' + resave_path
    os.system(command)

def collect_acc_data(folder):
    data_list = []
    for file in os.listdir(folder):
        if file[0] == 'a' and file[-1] == 't':
            data_list.append(folder + '/' + file)

    return sorted(data_list)

def get_acc_timestamp(path):
    return float(path.split('-')[1])

def read_acc_data(file_path):
    timestamp = get_acc_timestamp(file_path)
    file = open(file_path, 'r')
    data = file.read().split(',')
    for i in range(len(data)):
        data[i] = float(data[i])

    data.insert(0, timestamp)
    return data

def plot_acc_data(folder):
    acc_path = collect_acc_data(folder)
    x = []
    y = []
    z = []
    for path in acc_path:
        data = read_acc_data(path)
        x.append(data[1])
        y.append(data[2])
        z.append(data[3])
    plt.plot(x)
    plt.plot(y)
    plt.plot(z)
    plt.show()

def compute_acc_vel_pos(acc_data_1, acc_data_2, v_1, p_1):
    t1 = acc_data_1[0]
    t2 = acc_data_2[0]
    t_delta = t2 - t1

    acc_xyz_1 = np.array(acc_data_1[1:4])
    acc_xyz_2 = np.array(acc_data_2[1:4])
    a_avg = (acc_xyz_1 + acc_xyz_2) / 2.
    v_2 = v_1 + a_avg * t_delta
    p_2 = p_1 + v_1 * t_delta + a_avg * t_delta * t_delta / 2.

    # pdb.set_trace()
    return v_2, p_2

def plot_imu_traj(folder):
    acc_path = collect_acc_data(folder)

    p_x = []
    p_y = []
    p_z = []

    v_cur = np.array([0., 0., 0.])
    p_cur = np.array([0., 0., 0.])
    N = len(acc_path)
    for idx in range(N-1):
        p_x.append(p_cur[0])
        p_y.append(p_cur[1])
        p_z.append(p_cur[2])
        acc_1 = read_acc_data(acc_path[idx])
        acc_2 = read_acc_data(acc_path[idx + 1])
        v_cur, p_cur = compute_acc_vel_pos(acc_1, acc_2, v_cur, p_cur)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(p_x[:], p_y[:], p_z[:0])
    ax.plot(p_x[:-1], p_y[:-1], p_z[:-1])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    # plt.plot(p_x)
    # plt.plot(p_y)
    # plt.plot(p_z)
    # plt.show()

def read_amazon_data(data_file_name):
    data = open(data_file_name,"rb")
    rgbd = pickle.load(data)
    rgb = rgbd['rgb']

    rgb_new = np.zeros(rgb.shape, dtype=np.uint8)
    xx, yy = np.meshgrid(np.arange(0, 640, 1), np.arange(0, 480, 1), sparse=False)
    for i in range(0, 640, 1):
        for j in range(0, 480, 1):
            x_std, y_std = wrap_pixel(float(i), float(j))
            x_std = round(x_std)
            y_std = round(y_std)
            if x_std >= 0 and x_std <= (640 -1) and y_std >= 0 and y_std <= (480 -1):
                rgb_new[y_std, x_std, :] = rgb[j, i, :]

    plt.imshow(np.hstack((rgb, rgb_new)))
    plt.show(block=True)

def wrap_pixel(x, y):
    [fx_orig, fy_orig, cx_orig, cy_orig] = [400.516317, 400.410970, 320.171183, 243.274495]
    [fx_std, fy_std, cx_std, cy_std] = [518.857901, 519.469611, 325.582449, 253.736166]
    x_norm = (x - cx_orig) / fx_orig
    y_norm = (y - cy_orig) / fy_orig
    x_std = fx_std * x_norm + cx_std
    y_std = fy_std * y_norm + cy_std

    return x_std, y_std

def vis_depth(depth_map):
    vmax = np.percentile(depth_map, 95)
    vmin = depth_map.min()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='viridis')

    colormapped_im = (mapper.to_rgba(depth_map)[:, :, :3][:,:,::-1] * 255).astype(np.uint8)

    return colormapped_im[:, :, ::-1]

def vis_float_image(folder):
    seqs = sorted(os.listdir(folder))
    idx = 0
    for seq in seqs:
        if 'yml' in seq and idx >= 0:
            image_path = folder + '/' + seq
            # image = np.array(Image.open(image_path), dtype=np.float)
            fs = cv2.FileStorage(image_path, cv2.FILE_STORAGE_READ)
            image = fs.getNode('depth_map').mat()

            rgb_image = np.array(Image.open('/home/jiatianwu/eval/eval_data/' + seq[:-4] + '.jpg'))
            # image_bf = bilateral_filter(rgb_image, image)
            image_bf = image
            depth_image = vis_depth(image_bf)
            image_save = np.vstack((rgb_image, depth_image))
            Image.fromarray(image_save).save('/home/jiatianwu/eval/eval_comp/' + seq[:-4] + '.jpg')
            # plt.imshow(vis_depth(image))
            # plt.show()

        idx += 1

def vis_zcu_image(folder):
    seqs = sorted(os.listdir(folder))
    idx = 0
    for seq in seqs:
        data_path = folder + '/' + seq
        data = open(data_path,"rb")
        rgbd = pickle.load(data)

        rgb = rgbd['rgb']
        depth_image = rgbd['depth']
        depth_tflite_image = rgbd['tflite']
        depth_vitis_image = rgbd['vitis']
        image_show = np.vstack((rgb, depth_image, depth_tflite_image, depth_vitis_image))

        img = Image.fromarray(image_show)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("Agane.ttf", 32)
        draw.text((0, 0),"RGB Image",(255,0,0),font=font)
        draw.text((0, 480),"Depth Image before Quantization",(255,0,0),font=font)
        draw.text((0, 960),"Depth Image post-quantization \n EdgeTPU",(255,0,0),font=font)
        draw.text((0, 1440),"Depth Image post-quantization \n DPU",(255,0,0),font=font)
        img.save('saved_images/' + str(idx).zfill(6) + '.jpg')
        # plt.imshow(image_show)
        # plt.show(block=False)

        # plt.pause(0.001)
        # plt.clf()

        idx += 1

def process_tello(folder):
    seqs = sorted(os.listdir(folder))
    for idx in range(0, len(seqs), 5):
        data_path = folder + '/' + str(idx).zfill(6) + '.jpg'
        image = Image.open(data_path).resize((640, 480))
        image.save('/home/jiatianwu/dataset/tello_vga/' + str(int(idx/5)).zfill(6) + '.jpg')

        data_dict = {'rgb': np.array(image)}
        dict_file_name = '/home/jiatianwu/dataset/tello_pickle/' + str(int(idx/5)).zfill(6) + '.pkl'
        f = open(dict_file_name,"wb")
        pickle.dump(data_dict,f)
        f.close()

def vis_pickle_image(data_path):
    data = open(data_path,"rb")
    rgbd = pickle.load(data)

    rgb = rgbd['rgb']
    depth_image = rgbd['depth']
    depth_image_zcu = rgbd['tflite']
    image_show = np.vstack((rgb, depth_image, depth_image_zcu))

    img = Image.fromarray(image_show)
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("Agane.ttf", 32)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0, 0),"Sample Text",(255,0,0),font=font)
    draw.text((0, 480),"Sample Text",(255,0,0),font=font)
    draw.text((0, 960),"Sample Text",(255,0,0),font=font)
    img.save('sample-out.jpg')

    # plt.imshow(image_show)
    # plt.show(block=True)

if __name__ == "__main__":
    # resave_imu_data()
    # plot_acc_data('/home/jiatian/dataset/office_kitchen_0001a')
    # plot_imu_traj('/home/jiatian/dataset/office_kitchen_0001a')
    # read_amazon_data('/home/jiatian/dataset/amazon/raw_data/000001.pkl')
    # vis_float_image('/home/jiatianwu/eval/eval_res')
    vis_zcu_image('/home/jiatianwu/project/vitis-ai/nod_depth/saved_data/rgbd_tflite_vitis_data')
    # process_tello('/home/jiatianwu/dataset/tello')
    # vis_pickle_image('/home/jiatianwu/000001.pkl')