#Copyright 2020 Nod Labs
import argparse
import yaml
import sys
import os
import time
import h5py
import pickle
import numpy as np
import math
from scipy.io import matlab
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

from utils.bilateral_filter import bilateral_filter
from utils.tools import *
from depth_engine import DepthEngine

class App:

    def __init__(self, args):
        self.flag_exit = False
        self.halfres = args.halfres
        self.enable_filter = args.filter
        self.show_cover_ratio_only = args.show_cover_only
        self.is_save_data = args.save_data
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.eval = args.eval
        self.flip_lr = args.flip_lr
        self.flip_color = args.flip_color

        if self.is_save_data:
            os.makedirs('saved_data', exist_ok = True)
            os.makedirs(self.output_dir, exist_ok = True)

        if args.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]="0"

        config_path = 'config/noddepth_nyu_halfvga_eval.yml'

        datasource_path = '/home/jiatian/dataset/nyu_depth_v2_labeled.mat'
        self.setup_datasource(datasource_path)

        with open(config_path, 'r') as f:
            config = yaml.load(f)
        self.depth_engine = DepthEngine(config=config)
        self.depth_engine.load_model()

        self.output_size = (self.depth_engine.img_width, self.depth_engine.img_height) # (width, height)

        self.cr_gt = []
        self.cr_pred = []
        self.rmse_99 = []
        self.rmse_95 = []
        self.rmse_log = []
        self.abs_rel = []
        self.sq_rel = []
        self.scalar = []
        self.a1 = []
        self.a2 = []
        self.a3 = []

    def setup_datasource(self, path):
        hf = h5py.File(path, 'r')
        self.images_soure = np.array(hf.get('images'))
        self.depths_gt_source = np.array(hf.get('depths'))
        self.batch_size = self.depths_gt_source.shape[0]

    def get_data(self, idx=None):
        if idx < self.batch_size:
            image = np.transpose(self.images_soure[idx], (2, 1, 0))
            depth_gt = np.transpose(self.depths_gt_source[idx], (1, 0))

            return {"image" : image, "depth_gt": depth_gt}
        else:
            self.flag_exit = True
            return None

    def save_data(self, rgb, depth_pred, depth_gt=None, idx=None):
        data_dict = {'rgb': rgb, 'depth_pred': depth_pred, 'depth_gt':depth_gt}
        data_file_name = self.output_dir + '/' + str(idx).zfill(6) + '.pkl'
        f = open(data_file_name,"wb")
        pickle.dump(data_dict,f)
        f.close()

    def preprocess_image(self, image):
        if image is None:
            return None

        color_image = Image.fromarray(image)
        if color_image.size != self.output_size:
            color_image = color_image.resize(self.output_size)

        return np.array(color_image)
    
    def flip_image_color(self, image):
        return image[:, :, ::-1]

    def process_confidence_map(self, image):
        confidence_map = np.uint16(65536 - image * 1000)

        return confidence_map

    def get_metrics(self, depth_map, depth_map_gt):
        res_dict = eval_depth_nod(depth_map, depth_map_gt, self.depth_engine.min_depth, self.depth_engine.max_depth)
        s_gt_cover_ratio = 'GT depth cover ratio: ' + str(res_dict['gt_depth_cover_ratio']) + '%\n'
        s_pred_cover_ratio = 'Nod depth cover ratio: ' + str(res_dict['pred_depth_cover_ratio']) + '%\n'


        s_abs_rel = 'Absolute relative error: ' + '{:f}'.format(res_dict['abs_rel']) + '\n'
        s_sq_rel = 'Square relative error: ' + '{:f}'.format(res_dict['sq_rel']) + 'm\n'
        s_rms_99 = '99% Root mean squred error: ' + '{:f}'.format(res_dict['rms_99']) + 'm\n'
        s_rms_95 = '95% Root mean squred error: ' + '{:f}'.format(res_dict['rms_95']) + 'm\n'

        self.cr_gt.append(res_dict['gt_depth_cover_ratio'])
        self.cr_pred.append(res_dict['pred_depth_cover_ratio'])
        self.rmse_99.append(res_dict['rms_99'])
        self.rmse_95.append(res_dict['rms_95'])
        self.rmse_log.append(res_dict['rms_log'])
        self.abs_rel.append(res_dict['abs_rel'])
        self.sq_rel.append(res_dict['sq_rel'])
        self.scalar.append(res_dict['scalar'])
        self.a1.append(res_dict['a1'])
        self.a2.append(res_dict['a2'])
        self.a3.append(res_dict['a3'])

        if self.show_cover_ratio_only:
            s_viz = s_gt_cover_ratio + s_pred_cover_ratio
        else:
            s_viz = s_gt_cover_ratio + s_pred_cover_ratio + s_abs_rel + s_sq_rel + s_rms_99 + s_rms_95

        return s_viz

    def stats_print(self):
        print('-' * 60)
        print('A1:', np.mean(self.a1))
        print('A2:', np.mean(self.a2))
        print('A3:', np.mean(self.a3))
        print('RMSE LOG:', np.mean(self.rmse_log))
        print('RMSE 99%:', np.mean(self.rmse_99))
        print('RMSE 95%:', np.mean(self.rmse_95))
        print('ABS_REL:', np.mean(self.abs_rel))
        print('SQ_REL:', np.mean(self.sq_rel))
        print('SCALAR:', np.mean(self.scalar), '       ', np.std(self.scalar))
        print('CR_GT:', np.mean(self.cr_gt))
        print('CR_PRED:', np.mean(self.cr_pred))
        print('-' * 60)

    def handle_close(self, evt):
        self.flag_exit = True
        print('Close Nod Depth!')
        self.stats_print()
        sys.exit()

    def run(self):
        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', self.handle_close)
        idx = 0
        while not self.flag_exit:
            datasource = self.get_data(idx)
            if datasource is None:
                continue

            image = self.preprocess_image(datasource["image"])
            depth_gt = self.preprocess_image(datasource["depth_gt"])
            depth_gt_image = vis_depth(depth_gt)

            start = time.time()
            depth_pred = self.depth_engine.estimate_depth(image)
            print('Inference takes: ', time.time() - start)
            depth_pred_image = vis_depth(depth_pred)

            s_viz = self.get_metrics(depth_pred, depth_gt)
            toshow_image = np.hstack((image, depth_gt_image, depth_pred_image))
            plt.imshow(toshow_image)
            plt.text(0, -56, s_viz, fontsize=16)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.001)
            plt.clf()
            idx += 1
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run nod depth')
    parser.add_argument('--cpu',
                        default=False,
                        action='store_true',
                        help='use cpu to do inference')
    parser.add_argument('--halfres',
                        default=False,
                        action='store_true',
                        help='use full resolution 320*240')
    parser.add_argument('--eval',
                        default=False,
                        action='store_true',
                        help='run eval mode')
    parser.add_argument('--output_dir',
                        default='saved_data/raw_data',
                        type=str, help='output folder to save rgbd data got from kinect')
    parser.add_argument('--input_dir',
                        default='saved_data/raw_data',
                        type=str, help='input folder to eval rgbd data')
    parser.add_argument('--kinect_config',
                        default='config/kinect_config.json',
                        type=str, help='input json kinect config')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('--filter',
                        default=False,
                        action='store_true',
                        help='enable bilateral filter to refine depth map')
    parser.add_argument('--show_cover_only',
                        default=False,
                        action='store_true',
                        help='run retrain mode')
    parser.add_argument('--save_data',
                        default=False,
                        action='store_true',
                        help='save rgbd data during inferencing')
    parser.add_argument('--flip_lr',
                        default=False,
                        action='store_true',
                        help='flip left and right of rgb image')
    parser.add_argument('--flip_color',
                        default=False,
                        action='store_true',
                        help='flip rgb image color space')
    args = parser.parse_args()
    import tensorflow as tf
    print(tf.__version__)
    v = App(args)
    v.run()

    # Eval results
    # RMSE 99%: 0.5507934972873145
    # RMSE 95%: 0.47595251564074165
    # ABS_REL: 0.17021551190086803
    # SQ_REL: 0.13642955870286524
    # SCALAR: 0.002505729166737338         0.0006581630449547821
    # CR_GT: 99.0
    # CR_PRED: 99.0

# 640 * 480
# A1: 0.7209550857631701
# A2: 0.9404726814110307
# A3: 0.9850731714221016
# RMSE LOG: 0.096913084
# RMSE 99%: 0.5745349
# RMSE 95%: 0.49710596
# ABS_REL: 0.17999133
# SQ_REL: 0.14851524
# SCALAR: 2.7761567         0.7208286
# CR_GT: 99.0
# CR_PRED: 100.0

# 320 * 240
# A1: 0.6692187859443294
# A2: 0.9141683290919024
# A3: 0.9784690141908213
# RMSE LOG: 0.1072593
# RMSE 99%: 0.63563126
# RMSE 95%: 0.5608157
# ABS_REL: 0.20563722
# SQ_REL: 0.18119128
# SCALAR: 2.7390912         0.7006259
# CR_GT: 99.0
# CR_PRED: 100.0