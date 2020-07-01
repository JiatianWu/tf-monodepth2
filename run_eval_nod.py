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
        self.save_data = args.save_data
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.eval = args.eval
        self.flip_lr = args.flip_lr
        self.flip_color = args.flip_color

        if self.save_rgbd_data:
            os.makedirs('saved_data', exist_ok = True)
            os.makedirs(self.output_dir, exist_ok = True)

        if args.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"

        config_path = 'config/noddepth_nyu_eval.yml'

        datasource_path = '/home/nod/datasets/robot/20200611/rgbd_gt_data'
        self.setup_datasource(datasource_path, eval=False)

        with open(config_path, 'r') as f:
            config = yaml.load(f)
        self.depth_engine = DepthEngine(config=config)
        self.depth_engine.load_model()

        self.output_size = (self.depth_engine.img_width, self.depth_engine.img_height) # (width, height)
        self.crop_corner = (160, 0) # (x, y)
        self.crop_size = (960, 720) # (width, height)

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

    def setup_datasource(self, path, eval=False):
        self.folder_path = path
        self.batch_size = len(os.listdir(path))

    def get_datasource(self, idx=None):
        if idx < self.batch_size:
            data_path = self.folder_path + '/' + str(idx).zfill(6) + '.pkl'
            data = open(data_path,"rb")
            data_dict = pickle.load(data)
            image = data_dict['rgb']
            depth_gt = data_dict['depth_gt']

            return [image, depth_gt]
        else:
            self.flag_exit = True
            return None

    def save_rgbd_data(self, rgb, depth, depth_gt=None, idx=None):
        data_dict = {'rgb': rgb, 'depth_pred': depth, 'depth_gt':depth_gt}
        data_file_name = 'saved_data/tmp_data/' + str(idx).zfill(6) + '.pkl'
        f = open(data_file_name,"wb")
        pickle.dump(data_dict,f)
        f.close()

    def get_pred_depth(self, idx):
        data_path = self.folder_path + '/' + str(idx).zfill(6) + '.pkl'
        data = open(data_path,"rb")
        data_dict = pickle.load(data)
        depth = data_dict['depth_pred']

        return depth

    def preprocess_image(self, image):
        color_image = Image.fromarray(image)
        if color_image.size != self.output_size:
            color_image = color_image.crop(box=(self.crop_corner[0],
                                                self.crop_corner[1],
                                                self.crop_corner[0] + self.crop_size[0],
                                                self.crop_corner[1] + self.crop_size[1]))
            color_image = color_image.resize(self.output_size)
        
        debug = False
        if debug:
            color_image = (np.array(color_image, dtype=np.float) / 255 - 0.45) / 0.255

        return np.array(color_image)
    
    def flip_image_color(self, image):
        return image[:, :, ::-1]

    def process_confidence_map(self, image):
        confidence_map = np.uint16(65536 - image * 1000)

        return confidence_map

    def get_metrics(self, depth_map, depth_map_gt):
        #res_dict = eval_depth_nod(depth_map, depth_map_gt, self.depth_engine.min_depth, self.depth_engine.max_depth, 0.5977342578130507)
        res_dict = eval_depth_nod(depth_map, depth_map_gt, self.depth_engine.min_depth, self.depth_engine.max_depth, 1.0)
        s_gt_cover_ratio = 'Kinect depth cover ratio: ' + str(res_dict['gt_depth_cover_ratio']) + '%\n'
        s_pred_cover_ratio = 'Nod depth cover ratio: ' + str(res_dict['pred_depth_cover_ratio']) + '%\n'
        if self.show_cover_ratio_only:
            s_viz = s_gt_cover_ratio + s_pred_cover_ratio

            return s_viz

        s_abs_rel = 'Absolute relative error: ' + '{:f}'.format(res_dict['abs_rel']) + '\n'
        s_sq_rel = 'Square relative error: ' + '{:f}'.format(res_dict['sq_rel']) + 'm\n'
        s_rms_99 = '99% Root mean squred error: ' + '{:f}'.format(res_dict['rms_99']) + 'm\n'
        s_rms_95 = '95% Root mean squred error: ' + '{:f}'.format(res_dict['rms_95']) + 'm\n'
        s_viz = s_gt_cover_ratio + s_pred_cover_ratio + s_abs_rel + s_sq_rel + s_rms_99 + s_rms_95

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

        return s_viz

    def stats_print(self):
        print('----------------------------------------------------------------------------')
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
        print('----------------------------------------------------------------------------')

    def handle_close(self, evt):
        self.flag_exit = True
        print('Close Nod Depth!')
        sys.exit()

    def run(self):
        fig = plt.figure()
        fig.canvas.mpl_connect('close_event', self.handle_close)
        idx = 0
        while not self.flag_exit:
            datasource = self.get_datasource(idx)
            if datasource is None:
                continue

            rgb = datasource[0]
            depth = datasource[1]

            rgb_image = self.preprocess_image(rgb)
            depth_map_gt = self.preprocess_image(depth)
            depth_image_gt = vis_depth(depth_map_gt)

            start = time.time()
            if self.eval is False:
                if self.flip_lr:
                    rgb_image_lr = np.fliplr(rgb_image)

                if self.flip_color:
                    rgb_image_color = self.flip_image_color(rgb_image)

                start = time.time()
                depth_map = self.depth_engine.estimate_depth(rgb_image)

                if self.flip_lr:
                    depth_map_lr = np.fliplr(self.depth_engine.estimate_depth(rgb_image_lr))
                    depth_map = 0.5 * depth_map + 0.5 * depth_map_lr
                    depth_map_uncertainty = np.abs(depth_map - depth_map_lr)
                    depth_map_uncertainty = self.process_confidence_map(depth_map_uncertainty)
                elif self.flip_color:
                    depth_map_color = self.depth_engine.estimate_depth(rgb_image_color)
                    depth_map = 0.5 * depth_map + 0.5 * depth_map_color
                    depth_map_uncertainty = np.abs(depth_map - depth_map_color)
                    depth_map_uncertainty = self.process_confidence_map(depth_map_uncertainty)
                else:
                    depth_map_uncertainty = None

                print('Inference takes: ', time.time() - start)
                if self.enable_filter:
                    depth_map = bilateral_filter(rgb_image, depth_map, depth_map_uncertainty)
                    if math.isnan(depth_map.max()):
                        idx += 1
                        continue
            else:
                depth_map = self.get_pred_depth(idx)
            print(time.time() - start)
            depth_image = vis_depth(depth_map)

            if self.save_data:
                self.save_rgbd_data(rgb_image, depth_map, depth_map_gt, idx)

            toshow_image = np.hstack((rgb, depth_image, depth_image_gt))
            print(idx)

            Image.fromarray(toshow_image).save('saved_data/tmp/' + str(idx).zfill(6) + '.jpg')
            # SCALAR: 0.002505729166737338         0.0006581630449547821
            s_viz = self.get_metrics(depth_map, depth_map_gt)
            plt.imshow(toshow_image)
            plt.text(0, -56, s_viz, fontsize=16)
            plt.axis('off')
            plt.title('    Kinect Raw Input                                         Nod Depth                    Kinect Depth with hole completion', fontsize=20, loc='left')
            plt.show(block=False)

            plt.pause(0.001)
            # plt.savefig('/home/nod/datasets/nyudepthV2/eval_metrics/' + str(idx).zfill(6) + '.jpg')
            plt.clf()
            idx += 1

        plt.close()
        self.stats_print()

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