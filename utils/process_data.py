import os
import pdb
import h5py
import pickle
import numpy as np
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm
import tensorflow as tf

# from bilateral_filter import bilateral_filter
# from tools import *


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
    data = open(data_file_name, "rb")
    rgbd = pickle.load(data)
    rgb = rgbd['rgb']

    rgb_new = np.zeros(rgb.shape, dtype=np.uint8)
    xx, yy = np.meshgrid(np.arange(0, 640, 1),
                         np.arange(0, 480, 1), sparse=False)
    for i in range(0, 640, 1):
        for j in range(0, 480, 1):
            x_std, y_std = wrap_pixel(float(i), float(j))
            x_std = round(x_std)
            y_std = round(y_std)
            if x_std >= 0 and x_std <= (640 - 1) and y_std >= 0 and y_std <= (480 - 1):
                rgb_new[y_std, x_std, :] = rgb[j, i, :]

    plt.imshow(np.hstack((rgb, rgb_new)))
    plt.show(block=True)


def wrap_pixel(x, y):
    [fx_orig, fy_orig, cx_orig, cy_orig] = [
        400.516317, 400.410970, 320.171183, 243.274495]
    [fx_std, fy_std, cx_std, cy_std] = [
        518.857901, 519.469611, 325.582449, 253.736166]
    x_norm = (x - cx_orig) / fx_orig
    y_norm = (y - cy_orig) / fy_orig
    x_std = fx_std * x_norm + cx_std
    y_std = fy_std * y_norm + cy_std

    return x_std, y_std


def vis_depth(depth_map):
    vmax = np.percentile(depth_map, 90)
    vmin = depth_map.min()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='viridis')

    colormapped_im = (mapper.to_rgba(depth_map)[
                      :, :, :3][:, :, ::-1] * 255).astype(np.uint8)

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

            rgb_image = np.array(Image.open(
                '/home/jiatianwu/eval/eval_data/' + seq[:-4] + '.jpg'))
            # image_bf = bilateral_filter(rgb_image, image)
            image_bf = image
            depth_image = vis_depth(image_bf)
            image_save = np.vstack((rgb_image, depth_image))
            Image.fromarray(image_save).save(
                '/home/jiatianwu/eval/eval_comp/' + seq[:-4] + '.jpg')
            # plt.imshow(vis_depth(image))
            # plt.show()

        idx += 1


def vis_zcu_image(folder):
    seqs = sorted(os.listdir(folder))
    idx = 0
    for seq in seqs:
        data_path = folder + '/' + seq
        data = open(data_path, "rb")
        rgbd = pickle.load(data)

        rgb = rgbd['rgb']
        depth_image = rgbd['depth']
        depth_tflite_image = rgbd['tflite']
        depth_vitis_image = rgbd['vitis']
        image_show = np.vstack(
            (rgb, depth_image, depth_tflite_image, depth_vitis_image))

        img = Image.fromarray(image_show)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("Agane.ttf", 32)
        draw.text((0, 0), "RGB Image", (255, 0, 0), font=font)
        draw.text((0, 480), "Depth Image before Quantization",
                  (255, 0, 0), font=font)
        draw.text((0, 960), "Depth Image post-quantization \n EdgeTPU",
                  (255, 0, 0), font=font)
        draw.text((0, 1440), "Depth Image post-quantization \n DPU",
                  (255, 0, 0), font=font)
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
        image.save('/home/jiatianwu/dataset/tello_vga/' +
                   str(int(idx/5)).zfill(6) + '.jpg')

        data_dict = {'rgb': np.array(image)}
        dict_file_name = '/home/jiatianwu/dataset/tello_pickle/' + \
            str(int(idx/5)).zfill(6) + '.pkl'
        f = open(dict_file_name, "wb")
        pickle.dump(data_dict, f)
        f.close()


def vis_pickle_image(data_path):
    data = open(data_path, "rb")
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
    draw.text((0, 0), "Sample Text", (255, 0, 0), font=font)
    draw.text((0, 480), "Sample Text", (255, 0, 0), font=font)
    draw.text((0, 960), "Sample Text", (255, 0, 0), font=font)
    img.save('sample-out.jpg')

    # plt.imshow(image_show)
    # plt.show(block=True)


def vis_folder(folder):
    dirlist = sorted(os.listdir(folder))
    for seq in dirlist:
        image_path = folder + '/' + seq
        image = np.array(Image.open(image_path))

        plt.imshow(image)
        plt.show(block=True)
        plt.pause(0.001)
        plt.clf()

    plt.close()


def vis_image(path):
    image = Image.open(path)
    image = image.crop((0, 0, 1223, 699)).resize((608, 352))

    image = np.array(image)
    plt.imshow(image)
    plt.show(block=True)
    plt.close()


def vis_image_crop(path):
    image = Image.open(path)
    image = image.crop((0, 100, 1500, 900))

    image = np.array(image)
    plt.imshow(image)
    plt.show(block=True)
    plt.close()


def rename_folder(folder):
    dirlist = sorted(os.listdir(folder))
    for seq in dirlist:
        if '_CAM' in seq:
            continue
        else:
            os.rename(folder + '/' + seq, folder + '/' + seq + '_CAM_FRONT')


def rename_folder_util(folder):
    dirlist = sorted(os.listdir(folder))
    idx = 0
    for seq in dirlist:
        os.rename(folder + '/' + seq, folder +
                  '/' + str(idx).zfill(6) + '.jpg')
        idx += 1


def rename_folder_image(folder):
    dirlist = sorted(os.listdir(folder))
    idx = 0
    for seq in dirlist:
        Image.open(folder + '/' + seq).save(folder +
                                            '/' + str(idx).zfill(6) + '.jpg')
        idx += 1


def convert_rgb_folder(folder):
    dirlist = sorted(os.listdir(folder))
    for seq in dirlist:
        print('Processing: ', seq)
        imglist = sorted(os.listdir(folder + '/' + seq))
        for img in imglist:
            if '.jpg' in img:
                img_path = folder + '/' + seq + '/' + img
                Image.open(img_path).convert('RGB').save(img_path)


def plot_trajectory(data_file_name):
    # data = open(data_file_name,"rb")
    # poses_log = pickle.load(data)

    # poses_mat_log = []
    # import torch
    # for i in range(len(poses_log.keys())):
    #     pose = poses_log[i]
    #     pose = np.expand_dims(pose, axis=0)
    #     pose = np.expand_dims(pose, axis=0)
    #     pose_mat = transformation_from_parameters(torch.tensor(pose[:, :, :3]).float(), torch.tensor(pose[:, :, 3:]).float(), False)
    #     poses_mat_log.append(pose_mat.numpy())

    # xyzs = np.array(dump_xyz(poses_mat_log))
    # save_path = '/home/nod/datasets/kitti_eval_1/2011_09_26_drive_0022_sync_02/xyz_log.npy'
    # np.save(save_path, xyzs)

    xyzs = np.load(
        '/home/nod/datasets/kitti_eval_1/2011_09_26_drive_0022_sync_02/xyz_log.npy')

    xs = []
    ys = []
    zs = []
    for i in range(xyzs.shape[0]):
        xs.append(xyzs[i][0])
        ys.append(xyzs[i][1])
        zs.append(xyzs[i][2])

        plt.plot(xs, ys)
        plt.savefig(
            '/home/nod/datasets/kitti_eval_1/2011_09_26_drive_0022_sync_02/' + str(i).zfill(6) + '.jpg')


def vis_depth_pose(folder_depth, folder_pose):
    for i in range(792):
        depth_image = np.array(Image.open(
            folder_depth + '/' + str(i).zfill(6) + '.jpg'))
        pose_image = np.array(Image.open(
            folder_pose + '/' + str(i).zfill(6) + '.jpg'))

        tosave_image = np.vstack((depth_image, pose_image))
        Image.fromarray(tosave_image).save(
            '/home/nod/datasets/kitti_eval_1/save_images/' + '/' + str(i).zfill(6) + '.jpg')


def save_nyu_indoor_images(folder):
    dirlist = sorted(os.listdir(folder))
    step = 0
    for seq in dirlist:
        image_path = folder + '/' + seq
        image = Image.open(image_path).crop((0, 0, 640, 480))
        image.save('/home/nod/tmp/' + str(step).zfill(6) + '.jpg')
        step += 1


def viz_resize(folder):
    dirlist = sorted(os.listdir(folder))
    step = 0
    for seq in dirlist:
        image_path = folder + '/' + seq
        image = Image.open(image_path).resize((320, 240))
        image_dist = Image.open(image_path).resize((320, 200))

        plt.imshow(np.vstack((np.array(image), np.array(image_dist))))
        plt.show(block=True)


def read_process_nyu_data(path):
    hf = h5py.File(path, 'r')
    images = np.array(hf.get('images'))
    depths = np.array(hf.get('depths'))

    return images, depths

    batch_size = depths.shape[0]
    for idx in range(batch_size):
        image = images[idx]
        depth = depths[idx]

        image = np.transpose(image, (2, 1, 0))
        depth = np.transpose(depth, (1, 0))

        data_dict = {'rgb': image, 'depth_gt': depth}
        data_file_name = '/home/nod/datasets/nyudepthV2/rgbd_data_gt/' + \
            str(idx).zfill(6) + '.pkl'
        f = open(data_file_name, "wb")
        pickle.dump(data_dict, f)
        f.close()


def generate_pointcloud(rgb, depth, intrinsics=None, ply_file=None):
    points = []
    if intrinsics is not None:
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u, :]
            Z = depth[v, u]
            if Z == 0:
                continue
            if intrinsics is not None:
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy
            else:
                X = u
                Y = v
            points.append("%f %f %f %d %d %d 0\n" %
                          (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
                format ascii 1.0
                element vertex %d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                property uchar alpha
                end_header
                %s
                ''' % (len(points), "".join(points)))
    file.close()


def generate_pc_nyudepth(input_folder, output_folder):
    P_rect = np.eye(3, 3)
    P_rect[0, 0] = 5.1885790117450188e+02
    P_rect[0, 2] = 3.2558244941119034e+02
    P_rect[1, 1] = 5.1946961112127485e+02
    P_rect[1, 2] = 2.5373616633400465e+02

    build_output_dir(output_folder)
    dirlist = sorted(os.listdir(input_folder))
    step = 0
    for seq in dirlist:
        print('Processing idx: ', step)
        data_path = input_folder + '/' + seq
        data = open(data_path, "rb")
        data_dict = pickle.load(data)

        rgb = data_dict['rgb']
        depth_pred = data_dict['depth_pred'] * 0.002505729166737338
        depth_gt = data_dict['depth_gt']

        # pc_pred_path = output_folder + '/' + str(step).zfill(6) + '.ply'
        pc_gt_path = output_folder + '/' + str(step).zfill(6) + '_gt.ply'
        # generate_pointcloud(rgb, depth_pred, P_rect, ply_file=pc_pred_path)
        generate_pointcloud(rgb, depth_gt, P_rect, ply_file=pc_gt_path)
        step += 1


def generate_pc_media(input_folder, output_folder):
    build_output_dir(output_folder)
    dirlist = sorted(os.listdir(input_folder))
    step = 1
    for seq in dirlist:
        print('Processing idx: ', step)
        data_path = input_folder + '/' + seq
        data = open(data_path, "rb")
        data_dict = pickle.load(data)

        rgb = data_dict['rgb']
        depth_pred = data_dict['depth']

        pc_pred_path = output_folder + '/' + str(step).zfill(6) + '.ply'
        generate_pointcloud(rgb, depth_pred, intrinsics=None,
                            ply_file=pc_pred_path)
        step += 1


def generate_pc_media_intrinsics(input_folder, output_folder):
    P_rect = np.eye(3, 3)
    P_rect[0, 0] = 296.91973631
    P_rect[0, 2] = 321.97504478
    P_rect[1, 1] = 297.37056543
    P_rect[1, 2] = 225.25890346

    build_output_dir(output_folder)
    dirlist = sorted(os.listdir(input_folder))
    step = 0
    for seq in dirlist:
        print('Processing idx: ', step)
        data_path = input_folder + '/' + seq
        data = open(data_path, "rb")
        data_dict = pickle.load(data)

        rgb = data_dict['rgb']
        depth_pred = data_dict['depth']

        pc_pred_path = output_folder + '/' + str(step).zfill(6) + '.ply'
        generate_pointcloud(
            rgb, depth_pred, intrinsics=P_rect, ply_file=pc_pred_path)
        step += 1


def generate_pc_kinect(input_folder, output_folder):
    P_rect = np.eye(3, 3)
    P_rect[0, 0] = 400.516317
    P_rect[0, 2] = 320.171183
    P_rect[1, 1] = 400.410970
    P_rect[1, 2] = 243.274495

    # build_output_dir(output_folder)
    # dirlist = sorted(os.listdir(input_folder))
    # step = 0
    # for seq in dirlist:
    #     print('Processing idx: ', step)
    #     data_path = input_folder + '/' + seq
    #     data = open(data_path,"rb")
    #     data_dict = pickle.load(data)

    #     rgb = data_dict['rgb']
    #     depth_pred = data_dict['depth_pred'] * 1000
    #     depth_gt = data_dict['depth_gt']

    #     pc_pred_path = output_folder + '/' + str(step).zfill(6) + '.ply'
    #     pc_gt_path = output_folder + '/' + str(step).zfill(6) + '_gt.ply'
    #     generate_pointcloud(rgb, depth_pred, P_rect, ply_file=pc_pred_path)
    #     generate_pointcloud(rgb, depth_gt, P_rect, ply_file=pc_gt_path)
    #     step += 1

    rgb = np.array(Image.open('/home/nod/project/dso/build/sample/00068.jpg'))
    depth_pred = np.array(Image.open(
        '/home/nod/project/dso/build/sample/00023.png')) * 6.66
    pc_pred_path = '/home/nod/project/dso/build/sample/00068.ply'
    generate_pointcloud(rgb, depth_pred, intrinsics=P_rect,
                        ply_file=pc_pred_path)


def build_output_dir(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=False)
    except:
        os.makedirs(output_dir, exist_ok=True)

    return output_dir


def crop_folder(folder, output_dir):
    idx = 0
    dirlist = sorted(os.listdir(folder))
    for seq in dirlist:
        image_path = folder + '/' + seq
        image = Image.open(image_path)
        # kinect
        # image = image.crop((210, 110, 1700, 710))
        image = image.crop((0, 100, 1500, 900))

        # plt.imshow(image)
        # plt.show(block=True)
        # plt.pause(0.001)
        # plt.clf()
        image.save(output_dir + '/' + str(idx).zfill(6) + '.jpg')
        idx += 1
    # plt.close()


def readmat(filepath):
    annots = loadmat(filepath)
    import pdb
    pdb.set_trace()


def save_nyu_eval_image(filepath):
    f = h5py.File(filepath)
    for k, v in f.items():
        if k == 'images':
            image_source = np.array(v)

    batch_size = image_source.shape[0]
    for idx in range(batch_size):
        image = image_source[idx]
        image = np.transpose(image, (2, 1, 0))

        Image.fromarray(image).save(
            '/home/nod/datasets/nyudepthV2/eval_data/' + str(idx).zfill(6) + '.jpg')


def comp_nod_sgm(nod_folder, sgm_folder):
    nod_dirlist = sorted(os.listdir(nod_folder))
    sgm_dirlist = sorted(os.listdir(sgm_folder))
    batch_size = len(nod_dirlist)
    for idx in range(1900, batch_size):
        print('Processing ', idx)
        nod_image = np.array(Image.open(nod_folder + '/' + nod_dirlist[idx]))
        sgm_image = np.array(Image.open(sgm_folder + '/' + sgm_dirlist[idx]))

        nod_pred_image = nod_image[:, 640:, :]
        sgm_rgb_image = sgm_image[:, :1280, :]
        sgm_depth_image = sgm_image[:, 1280:, :]
        toshow_image = np.vstack(
            (sgm_rgb_image, np.hstack((nod_pred_image, sgm_depth_image))))

        img = Image.fromarray(toshow_image)
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype("Agane.ttf", 36)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((0, 0), "RGB Left", (255, 0, 0), font=font)
        draw.text((640, 0), "RGB Right", (255, 0, 0), font=font)
        draw.text((0, 480), "Nod Depth", (255, 0, 0), font=font)
        draw.text((640, 480), "SGBM", (255, 0, 0), font=font)
        img.save('/home/nod/datasets/weanhall/comp/' +
                 str(idx).zfill(6) + '.jpg')


def rename_folder_weanhall(folder):
    dirlist = sorted(os.listdir(folder))
    idx = 0
    for seq in dirlist:
        os.rename(folder + '/' + seq, folder +
                  '/' + str(idx).zfill(6) + '.jpg')
        idx += 1


def rename_folder_tum(folder):
    dirlist = sorted(os.listdir(folder))
    idx = 0
    for seq in dirlist:
        os.rename(folder + '/' + seq, folder +
                  '/' + str(idx).zfill(5) + '.jpg')
        idx += 1


def add_metrics_weanhall(nod_folder, sgm_folder, image_folder=None):
    path = image_folder
    image_left_path_list = []
    image_right_path_list = []
    image_list = sorted(os.listdir(path))
    for image in image_list:
        if 'left' in image:
            image_left_path_list.append(path + '/' + image)
        else:
            image_right_path_list.append(path + '/' + image)

    dirlist = sorted(os.listdir(nod_folder))
    batch_size = len(dirlist)
    cover_text = False
    for idx in range(1900, batch_size):
        nod_data_path = nod_folder + '/' + str(idx).zfill(6) + '.pkl'
        sgm_data_path = sgm_folder + '/' + str(idx).zfill(6) + '.pkl'

        nod_data = open(nod_data_path, "rb")
        nod_data_dict = pickle.load(nod_data)
        nod_depth = nod_data_dict['depth']

        sgm_data = open(sgm_data_path, "rb")
        sgm_data_dict = pickle.load(sgm_data)
        sgm_depth = sgm_data_dict['depth']

        image_left_path = image_left_path_list[idx]
        image_right_path = image_right_path_list[idx]
        image_left = np.array(Image.open(image_left_path))
        image_right = np.array(Image.open(image_right_path))

        nod_depth_image = vis_depth(nod_depth)
        sgm_depth_image = vis_depth(sgm_depth, percent=70)

        toshow_image = np.vstack((np.hstack((image_left, image_right)), np.hstack(
            (nod_depth_image, sgm_depth_image))))

        res_dict = eval_depth_nod(nod_depth, sgm_depth, 0.1, 10)

        s_gt_cover_ratio = 'SGM cover ratio: ' + \
            str(res_dict['gt_depth_cover_ratio']) + '%\n'
        s_pred_cover_ratio = 'NodDepth cover ratio: ' + \
            str(res_dict['pred_depth_cover_ratio']) + '%\n'

        s_abs_rel = 'Absolute relative error: ' + \
            '{:f}'.format(res_dict['abs_rel']) + '\n'
        s_sq_rel = 'Square relative error: ' + \
            '{:f}'.format(res_dict['sq_rel']) + 'm\n'
        s_rms_99 = '99% Root mean squred error: ' + \
            '{:f}'.format(res_dict['rms_99']) + 'm\n'
        s_rms_95 = '95% Root mean squred error: ' + \
            '{:f}'.format(res_dict['rms_95']) + 'm\n'
        s_viz = s_gt_cover_ratio + s_pred_cover_ratio + \
            s_abs_rel + s_sq_rel + s_rms_99 + s_rms_95

        img = Image.fromarray(toshow_image)
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        font = ImageFont.truetype("Agane.ttf", 36)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text((0, 0), "RGB Left", (255, 0, 0), font=font)
        draw.text((640, 0), "RGB Right", (255, 0, 0), font=font)
        draw.text((0, 480), "Nod Depth", (255, 0, 0), font=font)
        draw.text((640, 480), "SGBM", (255, 0, 0), font=font)

        plt.imshow(np.array(img))
        plt.text(-550, 200, s_viz, fontsize=14)
        plt.axis('off')
        # plt.title('    Kinect Raw Input                           Nod Depth                          Kinect Depth', fontsize=24, loc='left')
        plt.show(block=False)
        plt.pause(0.01)
        plt.savefig('/home/nod/datasets/weanhall/eval_metrics/' +
                    str(idx).zfill(6) + '.jpg')
        plt.clf()

        # image.save('/home/nod/datasets/weanhall/comp_metrics_v2/' + str(idx - 1900).zfill(6) + '.jpg')


def viz_rgbd(folder):
    dirlist = sorted(os.listdir(folder))
    for seq in dirlist:
        data_path = folder + '/' + seq
        data = open(data_path, "rb")
        data_dict = pickle.load(data)
        rgb = data_dict['rgb']
        depth = data_dict['depth_pred'] * 1000
        depth_gt = data_dict['depth_gt']

        # r = rgb[:, :, 0]
        toshow_image = np.hstack((depth, depth_gt))
        plt.imshow(toshow_image)
        plt.show(block=True)


def save_rgb(folder):
    dirlist = sorted(os.listdir(folder))
    idx = 0
    for seq in dirlist:
        data_path = folder + '/' + seq
        data = open(data_path, "rb")
        data_dict = pickle.load(data)
        rgb = data_dict['rgb']

        Image.fromarray(rgb).save(
            '/home/nod/datasets/robot/20200611/dso/images/' + str(idx).zfill(5) + '.jpg')
        idx += 1


def save_depth(folder):
    dirlist = sorted(os.listdir(folder))
    idx = 0
    for seq in dirlist:
        data_path = folder + '/' + seq
        data = open(data_path, "rb")
        data_dict = pickle.load(data)
        depth = np.uint16(data_dict['depth_pred'] * 1000)

        Image.fromarray(depth).save(
            '/home/nod/tmp/' + str(idx).zfill(5) + '.png')
        idx += 1


def process_kitti_data(folder):
    dirlist = sorted(os.listdir(folder))
    for seq in dirlist:
        seq_path = folder + '/' + seq
        imglist = sorted(os.listdir(seq_path))
        for file in imglist:
            if '.jpg' in file:
                image_path = seq_path + '/' + file
                image = np.array(Image.open(image_path))[:, 640:1280, :]
                Image.fromarray(image).save(image_path)


def merge_pickle_data(folder_1, folder_2):
    dirlist = sorted(os.listdir(folder_1))
    for file in dirlist:
        data_path_1 = folder_1 + '/' + file
        data_path_2 = folder_2 + '/' + file

        data_1 = open(data_path_1, "rb")
        data_dict_1 = pickle.load(data_1)

        rgb = data_dict_1['rgb']
        depth_pred = data_dict_1['depth']

        data_2 = open(data_path_2, "rb")
        data_dict_2 = pickle.load(data_2)
        depth_gt = data_dict_2['depth_gt'][:, 640:1280]

        data_dict = {'rgb': rgb,
                     'depth_pred': depth_pred,
                     'depth_gt': depth_gt}

        dict_file_name = '/home/nod/datasets/robot/20200611/rgbd_gt_data/' + file
        f = open(dict_file_name, "wb")
        pickle.dump(data_dict, f)
        f.close()


def flip_array(image_path):
    image = np.array(Image.open(image_path))
    image_flip_lr = np.fliplr(image)

    plt.imshow(np.vstack((image, image_flip_lr)))
    plt.show(block=True)


def read_confidence(image_path):
    image = np.array(Image.open(image_path))


def read_depth(image_path):
    image = np.array(Image.open(image_path))
    # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    import pdb
    pdb.set_trace()


def process_dso(data_path):
    data = open(data_path, "rb")
    data_dict = pickle.load(data)

    rgb = data_dict['rgb']
    depth_pred = data_dict['depth_pred'] * 1000
    depth_gt = data_dict['depth_gt']

    depth_dso = np.array(Image.open(
        '/home/nod/project/dso/build/sample/00023.png'))

    # toshow_image = np.hstack((depth_dso, depth_gt))
    # plt.imshow(toshow_image)
    # plt.show(block = True)

    # Scalar matching, scalar is 4.257004157652051
    # scalar_dso
    # 4.257004157652051
    # scalar_pred
    # 0.7131490173152352
    mask_gt = np.logical_and(depth_gt > 10, depth_gt < 2000)
    mask_pred = np.logical_and(depth_pred > 100, depth_pred < 10000)
    mask_dso = np.logical_and(depth_dso > 10, depth_dso < 2000)
    scalar_dso = np.mean(depth_gt[mask_gt]) / np.mean(depth_dso[mask_dso])
    scalar_pred = np.mean(depth_gt[mask_gt]) / np.mean(depth_pred[mask_pred])

    # depth_dso_image = vis_depth(depth_dso * scalar_dso)
    # depth_pred_image = vis_depth(depth_pred * scalar_pred)
    # Image.fromarray(depth_dso_image).save('/home/nod/project/dso/build/sample/depth_dso.png')
    # Image.fromarray(depth_pred_image).save('/home/nod/project/dso/build/sample/depth_pred.png')


def eval_densify(data_path):
    data = open(data_path, "rb")
    data_dict = pickle.load(data)

    # pdb.set_trace()
    rgb = data_dict['rgb']
    depth_pred = data_dict['depth_pred']
    depth_dso = data_dict['depth_dso']
    depth_gt = data_dict['depth_gt']
    depth_densify = data_dict['depth_densify']

    depth_pred_image = vis_depth(depth_pred)
    depth_dso_image = vis_depth(depth_dso)
    depth_densify_image = vis_depth(depth_densify)
    depth_gt_image = vis_depth(depth_gt)
    # plt.title('    Nod Depth                             DSO Depth                          Densified DSO                Kinect Depth', fontsize=20, loc='left')
    # plt.imshow(np.hstack((depth_pred_image, depth_dso_image, depth_densify_image, depth_gt_image)))
    plt.title('    Nod Depth                  Densified DSO              Kinect Depth',
              fontsize=12, loc='left')
    plt.imshow(np.hstack((depth_pred, depth_densify, depth_gt)))
    plt.show(block=True)

    min_depth = 0.1
    max_depth = 10
    print('------------------EVAL Depth Model------------------')
    eval_pred_dict = eval_depth_nod(
        depth_pred, depth_gt, min_depth, max_depth, 1.0)
    print_eval_dict(eval_pred_dict)
    print('------------------EVAL DSO------------------')
    eval_dso_dict = eval_depth_nod(
        depth_dso, depth_gt, min_depth, max_depth, 1.0)
    print_eval_dict(eval_dso_dict)
    print('------------------EVAL Densified DSO------------------')
    eval_densify_dict = eval_depth_nod(
        depth_densify, depth_gt, min_depth, max_depth, 1.0)
    print_eval_dict(eval_densify_dict)


def vis_rgbd_pickle_image(folder_1, folder_2):
    dirlist_1 = sorted(os.listdir(folder_1))
    dirlist_2 = sorted(os.listdir(folder_2))
    for idx in range(len(dirlist_1)):
        datapath_1 = open(folder_1 + '/' + dirlist_1[idx], "rb")
        data_1 = pickle.load(datapath_1)

        rgb = data_1['rgb']
        depth_image_tpu = vis_depth(data_1['depth_pred'])
        depth_image_gt = vis_depth(data_1['depth_gt'])

        datapath_2 = open(folder_2 + '/' + dirlist_2[idx], "rb")
        depth_image_dpu = np.array(Image.open(datapath_2))[480:, :, :]
        image_show = np.hstack(
            (rgb, depth_image_gt, depth_image_dpu, depth_image_tpu))

        img = Image.fromarray(image_show)
        img.save('/home/nod/datasets/nyudepthV2/rgbd_tpu_dpu/' +
                 dirlist_2[idx])


def convert_16bit_rgb(folder_1, folder_2):
    dirlist_1 = sorted(os.listdir(folder_1))
    for idx in range(len(dirlist_1)):
        datapath_1 = folder_1 + '/' + dirlist_1[idx]
        image_1 = plt.imread(datapath_1)
        plt.imsave(folder_2 + '/' + dirlist_1[idx], image_1)
        print(idx)


def resize_folder(folder):
    dirlist = sorted(os.listdir(folder))
    for idx in range(len(dirlist)):
        datapath = folder + '/' + dirlist[idx]
        Image.open(datapath).resize((896, 896)).save(
            '/home/jiatian/project/Zooming-Slow-Mo-CVPR-2020/tmp' + '/' + dirlist[idx])
        print(idx)


if __name__ == "__main__":
    # resave_imu_data()
    # plot_acc_data('/home/jiatian/dataset/office_kitchen_0001a')
    # plot_imu_traj('/home/jiatian/dataset/office_kitchen_0001a')
    # read_amazon_data('/home/jiatian/dataset/amazon/raw_data/000001.pkl')
    # vis_float_image('/home/jiatianwu/eval/eval_res')
    # vis_zcu_image('/home/jiatianwu/project/vitis-ai/nod_depth/saved_data/rgbd_tflite_vitis_data')
    # process_tello('/home/jiatianwu/dataset/tello')
    # vis_pickle_image('/home/jiatianwu/000001.pkl')
    # vis_folder(folder='/home/nod/lyft_kitti/train/image_2')
    # vis_image('/home/nod/datasets/lyft/raw_data/000000/000000_9ccf7db5e9d2ab8847906a7f086aa7c0c189efecfe381d9120bf02c7de6907b9.png')
    # rename_folder_util('/home/nod/tmp_2')
    # plot_trajectory('/home/nod/datasets/kitti_eval_1/2011_09_26_drive_0022_sync_02/poses_log.pickle')
    # vis_depth_pose('/home/nod/datasets/kitti_eval/2011_09_26_drive_0022_sync_02', '/home/nod/datasets/kitti_eval_1/2011_09_26_drive_0022_sync_02/')
    # save_nyu_indoor_images('/home/nod/nod/nod/src/apps/nod_depth/saved_data/indoor_eval_res')
    # viz_resize('/home/nod/datasets/nod/images0')
    # read_process_nyu_data('/home/nod/datasets/nyudepthV2/nyu_depth_v2_labeled.mat')
    # generate_pc_nyudepth('/home/nod/datasets/nyudepthV2/rgbd_gt_data', '/home/nod/datasets/nyudepthV2/pc_gt')
    # crop_folder('/home/nod/tmp', '/home/nod/tmp_2')
    # readmat('/home/nod/Downloads/splits.mat')
    # save_nyu_eval_image('/home/nod/datasets/nyudepthV2/nyu_depth_v2_labeled.mat')
    # comp_nod_sgm('/home/nod/datasets/weanhall/eval', '/home/nod/datasets/weanhall/eval_sgm')
    # rename_folder_tum('/home/nod/datasets/robot/20200611/images')
    # add_metrics_weanhall('/home/nod/datasets/weanhall/rgbd_data_wean', '/home/nod/datasets/weanhall/eval_sgm_data', '/home/nod/datasets/weanhall/comp')
    # generate_pc_media_intrinsics('/home/nod/datasets/media/eval/eval_res_data', '/home/nod/datasets/media/eval/pc')
    # viz_rgbd('/home/nod/datasets/robot/20200611/rgbd_gt_data_finetune')
    # vis_image_crop('/home/nod/datasets/weanhall/eval_metrics/001907.jpg')
    # crop_folder('/home/nod/datasets/weanhall/eval_metrics', '/home/nod/datasets/weanhall/eval_metrics_crop')
    # add_metrics_weanhall('/home/nod/datasets/weanhall/eval_model_data', '/home/nod/datasets/weanhall/eval_sgm_data', '/home/nod/datasets/weanhall/rectified')
    # process_kitti_data('/home/nod/datasets/kitti/kitti_data')
    # rename_folder_image('/home/nod/datasets/nyudepthV2/test_kitchen/color')
    # merge_pickle_data('/home/nod/datasets/nyudepthV2/rgbd_data', '/home/nod/datasets/nyudepthV2/rgb_gt_data')
    # flip_array('/home/nod/datasets/nyudepthV2/eval_data/000000.jpg')
    # read_confidence('/home/nod/project/The_Bilateral_Solver/build/confidence.png')
    # merge_pickle_data('/home/nod/project/tf-monodepth2/saved_data/tmp_data', '/home/nod/datasets/robot/20200611/depth_data')
    # generate_pc_kinect('/home/nod/datasets/robot/20200611/rgbd_gt_data_finetune', '/home/nod/datasets/robot/20200611/pc_data_finetune')
    # save_rgb('/home/nod/datasets/robot/20200611/rgbd_gt_data')
    # save_depth('/home/nod/datasets/robot/20200611/rgbd_gt_data_finetune')
    # read_depth('/home/nod/project/dso/build/depths_out/00007.png')
    # process_dso('/home/nod/project/dso/build/sample/000068.pkl')
    # eval_densify('/home/nod/project/dso/build/sample/00068_densify.pkl')
    # convert_rgb_folder('/home/jiatian/dataset/tum')
    #vis_rgbd_pickle_image("/home/nod/datasets/nyudepthV2/rgbd_gt_tpu_nopp_data", "/home/nod/project/vitis-ai/mpsoc/vitis-ai-tool-example/tf_eval_script/eval_res")
    #convert_16bit_rgb('/home/jiatian/dataset/bu_tiv/lab1-test-seq1-red/red/TIV', '/home/jiatian/dataset/bu_tiv/lab1-test-seq1-red/red/rgb')
    resize_folder(
        '/home/jiatian/project/Zooming-Slow-Mo-CVPR-2020/test_example/adas')
