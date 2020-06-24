from __future__ import division
import argparse
import scipy.misc
import numpy as np
from PIL import Image
from glob import glob
from joblib import Parallel, delayed
import os

parser = argparse.ArgumentParser()

dataset_name = 'redwood'

if dataset_name == 'redwood':
    parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
    parser.add_argument("--img_height", type=int, default=480, help="image height")
    parser.add_argument("--img_width", type=int, default=640, help="image width")
    parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use")
    parser.add_argument("--dataset_dir", type=str, default='/home/jiatian/Downloads/redwood/', help="where the dataset is stored")
    parser.add_argument("--dataset_name", type=str, default='redwood', choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes", "tum", "tello", "nyu"])
    parser.add_argument("--dump_root", type=str, default='/home/jiatian/dataset/redwood/', help="Where to dump the data")

if dataset_name == 'nyu_fullRes':
    parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
    parser.add_argument("--img_height", type=int, default=480, help="image height")
    parser.add_argument("--img_width", type=int, default=640, help="image width")
    parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use")
    parser.add_argument("--dataset_dir", type=str, default='/freezer/nyudepthV2_raw/', help="where the dataset is stored")
    parser.add_argument("--dataset_name", type=str, default='nyu_fullRes', choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes", "tum", "tello", "nyu"])
    parser.add_argument("--dump_root", type=str, default='/home/jiatian/dataset/nyu_fullRes/', help="Where to dump the data")

if dataset_name == 'nyu': 
    parser.add_argument("--dataset_dir", type=str, default='/freezer/nyudepthV2_raw/', help="where the dataset is stored")
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

if dataset_name == 'lyft':
    parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
    parser.add_argument("--img_height", type=int, default=352, help="image height")
    parser.add_argument("--img_width", type=int, default=608, help="image width")
    parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use")
    parser.add_argument("--dataset_dir", type=str, default='/home/nod/datasets/lyft/raw_data/', help="where the dataset is stored")
    parser.add_argument("--dataset_name", type=str, default='lyft', choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes", "tum", "tello", "nyu"])
    parser.add_argument("--dump_root", type=str, default='/home/nod/datasets/lyft/train_data/', help="Where to dump the data")

if dataset_name == 'nod':
    parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
    parser.add_argument("--img_height", type=int, default=480, help="image height")
    parser.add_argument("--img_width", type=int, default=640, help="image width")
    parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use")
    parser.add_argument("--dataset_dir", type=str, default='/home/jiatian/dataset/nod_device/images0/', help="where the dataset is stored")
    parser.add_argument("--dataset_name", type=str, default='nod', choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes", "tum", "tello", "nyu"])
    parser.add_argument("--dump_root", type=str, default='/home/jiatian/dataset/nod_device/images0_train_data/', help="Where to dump the data")

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

    Image.fromarray(image_seq.astype(np.uint8)).save(dump_img_file)
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
        sequences_number = len(os.listdir(args.dataset_dir))
        for id in range(sequences_number):
            try:
                data_loader = nyu_loader(args.dataset_dir,
                                    split='sequence',
                                    sequence_id=id,
                                    img_height=args.img_height,
                                    img_width=args.img_width,
                                    seq_length=args.seq_length)
                # Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))
            except:
                print("ERROR in " + str(id))

    if args.dataset_name == 'nyu_fullRes':
        from nyu.nyu_loader import nyu_loader
        sequences_number = len(os.listdir(args.dataset_dir))
        for id in range(451, sequences_number):
            print('Sequence id: ', id)
            data_loader = nyu_loader(args.dataset_dir,
                                split='sequence',
                                sequence_id=id,
                                img_height=args.img_height,
                                img_width=args.img_width,
                                seq_length=args.seq_length)
            try:
                data_loader = nyu_loader(args.dataset_dir,
                                    split='sequence',
                                    sequence_id=id,
                                    img_height=args.img_height,
                                    img_width=args.img_width,
                                    seq_length=args.seq_length)
                Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))
            except:
                print("ERROR in " + str(id))

    if args.dataset_name == 'lyft':
        from lyft.lyft_loader import lyft_loader
        sequences_number = len(os.listdir(args.dataset_dir))
        for id in range(sequences_number):
            print('Sequence id: ', id)
            # data_loader = lyft_loader(args.dataset_dir,
            #                     split='sequence',
            #                     sequence_id=id,
            #                     img_height=args.img_height,
            #                     img_width=args.img_width,
            #                     seq_length=args.seq_length)
            # Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))
            try:
                data_loader = lyft_loader(args.dataset_dir,
                                    split='sequence',
                                    sequence_id=id,
                                    img_height=args.img_height,
                                    img_width=args.img_width,
                                    seq_length=args.seq_length)
                Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))
            except:
                print("ERROR in " + str(id))

    if args.dataset_name == 'nod':
        from nod.nod_loader import nod_loader
        sequences_number = len(os.listdir(args.dataset_dir))
        for id in range(sequences_number):
            print('Sequence id: ', id)
            try:
                data_loader = nod_loader(args.dataset_dir,
                                    split='sequence',
                                    sequence_id=id,
                                    img_height=args.img_height,
                                    img_width=args.img_width,
                                    seq_length=args.seq_length)
                Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))
            except:
                print("ERROR in " + str(id))

    if args.dataset_name == 'redwood':
        from redwood.redwood_loader import redwood_loader
        sequences_number = len(os.listdir(args.dataset_dir))
        for id in range(sequences_number):
            print('Sequence id: ', id)
            try:
                data_loader = redwood_loader(args.dataset_dir,
                                            split='sequence',
                                            sequence_id=id,
                                            img_height=args.img_height,
                                            img_width=args.img_width,
                                            seq_length=args.seq_length)
                Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n, args) for n in range(data_loader.num_train))
            except:
                print("ERROR in " + str(id))

    # Split into train/val
    np.random.seed(8964)
    subfolders = os.listdir(args.dump_root)
    with open(args.dump_root + 'train.txt', 'w') as tf:
        with open(args.dump_root + 'val.txt', 'w') as vf:
            for s in subfolders:
                # if '_CAM_FRONT' in s and 'LEFT' not in s and 'RIGHT' not in s:
                if not os.path.isdir(args.dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(args.dump_root, s, '*.jpg'))
                if args.dataset_name == 'nyu' or args.dataset_name == 'nyu_fullRes':
                    frame_ids = [os.path.basename(fi)[:-4] for fi in imfiles]
                else:
                    frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))

main()