import os
import pdb
import h5py
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

def rename_folder(folder):
    dirlist = sorted(os.listdir(folder))
    for seq in dirlist:
        if '_CAM' in seq:
            continue
        else:
            os.rename(folder + '/' + seq, folder + '/' + seq + '_CAM_FRONT')

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

    xyzs = np.load('/home/nod/datasets/kitti_eval_1/2011_09_26_drive_0022_sync_02/xyz_log.npy')

    xs = []
    ys = []
    zs = []
    for i in range(xyzs.shape[0]):
        xs.append(xyzs[i][0])
        ys.append(xyzs[i][1])
        zs.append(xyzs[i][2])

        plt.plot(xs, ys)
        plt.savefig('/home/nod/datasets/kitti_eval_1/2011_09_26_drive_0022_sync_02/' + str(i).zfill(6) + '.jpg')


def vis_depth_pose(folder_depth, folder_pose):
    for i in range(792):
        depth_image = np.array(Image.open(folder_depth + '/' + str(i).zfill(6) + '.jpg'))
        pose_image = np.array(Image.open(folder_pose + '/' + str(i).zfill(6) + '.jpg'))

        tosave_image = np.vstack((depth_image, pose_image))
        Image.fromarray(tosave_image).save('/home/nod/datasets/kitti_eval_1/save_images/' + '/' + str(i).zfill(6) + '.jpg')

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

def generate_pointcloud(rgb, depth, intrinsics, ply_file=None):
    points = []
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u, :]
            Z = depth[v, u]
            if Z==0:
                continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(ply_file,"w")
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
                '''%(len(points),"".join(points)))
    file.close()

def generate_pc_nyudepth(input_folder, output_folder):
    P_rect = np.eye(3, 3)
    P_rect[0,0] = 5.1885790117450188e+02
    P_rect[0,2] = 3.2558244941119034e+02
    P_rect[1,1] = 5.1946961112127485e+02
    P_rect[1,2] = 2.5373616633400465e+02

    build_output_dir(output_folder)
    dirlist = sorted(os.listdir(input_folder))
    step = 0
    for seq in dirlist:
        print('Processing idx: ', step)
        data_path = input_folder + '/' + seq
        data = open(data_path,"rb")
        data_dict = pickle.load(data)

        rgb = data_dict['rgb']
        depth_pred = data_dict['depth_pred'] * 2.82
        depth_gt = data_dict['depth_gt']

        pc_pred_path = output_folder + '/' + str(step).zfill(6) + '.ply'
        pc_gt_path = output_folder + '/' + str(step).zfill(6) + '_gt.ply'
        pc_pred = generate_pointcloud(rgb, depth_pred, P_rect, ply_file=pc_pred_path)
        pc_gt = generate_pointcloud(rgb, depth_gt, P_rect, ply_file=pc_gt_path)
        step += 1

def build_output_dir(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=False)
    except:
        os.makedirs(output_dir, exist_ok=True)

    return output_dir

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
    # rename_folder('/home/nod/datasets/lyft/raw_data')
    # plot_trajectory('/home/nod/datasets/kitti_eval_1/2011_09_26_drive_0022_sync_02/poses_log.pickle')
    # vis_depth_pose('/home/nod/datasets/kitti_eval/2011_09_26_drive_0022_sync_02', '/home/nod/datasets/kitti_eval_1/2011_09_26_drive_0022_sync_02/')
    # save_nyu_indoor_images('/home/nod/nod/nod/src/apps/nod_depth/saved_data/indoor_eval_res')
    # viz_resize('/home/nod/datasets/nod/images0')
    # read_process_nyu_data('/home/nod/datasets/nyudepthV2/nyu_depth_v2_labeled.mat')
    generate_pc_nyudepth('/home/nod/datasets/nyudepthV2/eval_res_data_gray', '/home/nod/datasets/nyudepthV2/eval_res_pc_gray')