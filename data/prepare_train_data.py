from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os

parser = argparse.ArgumentParser()

dataset_name = 'nyu'

parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=240, help="image height")
parser.add_argument("--img_width", type=int, default=320, help="image width")
parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use")

if dataset_name == 'nyu': 
    parser.add_argument("--dataset_dir", type=str, default='/freezer/nyudepthV2_raw', help="where the dataset is stored")
    parser.add_argument("--dataset_name", type=str, default='nyu', choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes", "tum", "tello", "nyu"])
    parser.add_argument("--dump_root", type=str, default='/home/jiatian/dataset/nyu/', help="Where to dump the data")

if dataset_name == 'tum': 
    parser.add_argument("--dataset_dir", type=str, default='/home/jiatian/dataset/all_sequences/', help="where the dataset is stored")
    parser.add_argument("--dataset_name", type=str, default='tum', choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes", "tum", "tello"])
    parser.add_argument("--dump_root", type=str, default='/home/jiatian/dataset/tum/', help="Where to dump the data")

if dataset_name == 'tello': 
    parser.add_argument("--dataset_dir", type=str, default='/home/jiatian/dataset/tello_raw/', help="where the dataset is stored")
    parser.add_argument("--dataset_name", type=str, default='tello', choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes", "tum", "tello"])
    parser.add_argument("--dump_root", type=str, default='/home/jiatian/dataset/tello/', help="Where to dump the data")

args = parser.parse_args()

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def dump_example(n, args):
    if n % 2000 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    dump_dir = os.path.join(args.dump_root, example['folder_name'])
    # if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
    try: 
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader
    if args.dataset_name == 'kitti_odom':
        from kitti.kitti_odom_loader import kitti_odom_loader
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_eigen':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='eigen',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_stereo':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='stereo',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)

    if args.dataset_name == 'cityscapes':
        from cityscapes.cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.dataset_name == 'tum':
        from tum.tum_loader import tum_loader
        id = 0
        data_loader = tum_loader(args.dataset_dir,
                            split='sequence',
                            sequence_id=id,
                            img_height=args.img_height,
                            img_width=args.img_width,
                            seq_length=args.seq_length)
        # Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))

    if args.dataset_name == 'tello':
        from tello.tello_loader import tello_loader
        data_loader = tello_loader(args.dataset_dir,
                                   split='pics',
                                   img_height=args.img_height,
                                   img_width=args.img_width,
                                   seq_length=args.seq_length)

    if args.dataset_name == 'nyu':
        from nyu.nyu_loader import nyu_loader
        for id in range(583):
            data_loader = nyu_loader(args.dataset_dir,
                                split='sequence',
                                sequence_id=id,
                                img_height=args.img_height,
                                img_width=args.img_width,
                                seq_length=args.seq_length)
            Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))

    # import pdb; pdb.set_trace()
    # Split into train/val
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    with open(args.dump_root + 'train.txt', 'w') as tf:
        with open(args.dump_root + 'val.txt', 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))

main()

