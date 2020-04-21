#Copyright 2020 Nod Labs
import argparse
import yaml
import sys
import os
import time
import h5py
import pickle
import numpy as np
from scipy.io import matlab
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

from bilateral_filter import bilateral_filter
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

        if self.save_rgbd_data:
            os.makedirs('saved_data', exist_ok = True)
            os.makedirs(self.output_dir, exist_ok = True)

        if args.cpu:
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"

        datasource_path = '/home/nod/datasets/nyudepthV2/nyu_depth_v2_labeled.mat'
        config_path = 'config/model_eval.yml'

        self.setup_datasource(datasource_path)

        with open(config_path, 'r') as f:
            config = yaml.load(f)
        self.depth_engine = DepthEngine(config=config)
        self.depth_engine.load_model()

        self.output_size = (self.depth_engine.img_width, self.depth_engine.img_height) # (width, height)
        self.crop_corner = (160, 0) # (x, y)
        self.crop_size = (960, 720) # (width, height)

        self.rmse = []
        self.abs_rel = []
        self.sq_rel = []
        self.scalar = []

    def setup_datasource(self, path):
        f = h5py.File(path)
        for k, v in f.items():
            if k == 'depths':
                self.depth_source = np.array(v)

            if k == 'images':
                self.image_source = np.array(v)

        assert(self.depth_source.shape[0] == self.image_source.shape[0])
        self.batch_size = self.depth_source.shape[0]

    def get_datasource(self, idx=None):
        if idx < self.batch_size:
            image = self.image_source[idx]
            depth = self.depth_source[idx]

            image = np.transpose(image, (2, 1, 0))
            depth = np.transpose(depth, (1, 0))

            return [image, depth]
        else:
            self.flag_exit = True
            return None

    def save_rgbd_data(self, rgb, depth, idx):
        data_dict = {'rgb': rgb, 'depth': depth}
        data_file_name = 'saved_data/rgbd_data/' + str(idx).zfill(6) + '.pkl'
        f = open(data_file_name,"wb")
        pickle.dump(data_dict,f)
        f.close()

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

    def get_metrics(self, depth_map, depth_map_gt):
        res_dict = eval_depth(depth_map, depth_map_gt, self.depth_engine.min_depth, self.depth_engine.max_depth)
        s_gt_cover_ratio = 'Kinect depth cover ratio: ' + str(res_dict['gt_depth_cover_ratio']) + '%\n'
        s_pred_cover_ratio = 'Nod depth cover ratio: ' + str(res_dict['pred_depth_cover_ratio']) + '%\n'
        if self.show_cover_ratio_only:
            s_viz = s_gt_cover_ratio + s_pred_cover_ratio

            return s_viz

        s_abs_rel = 'Absolute relative error: ' + '{:f}'.format(res_dict['abs_rel']) + '\n'
        s_sq_rel = 'Square relative error: ' + '{:f}'.format(res_dict['sq_rel']) + '\n'
        s_rms = 'Root mean squred error: ' + '{:f}'.format(res_dict['rms']) + '\n'
        s_viz = s_gt_cover_ratio + s_pred_cover_ratio + s_abs_rel + s_sq_rel + s_rms

        self.rmse.append(res_dict['rms'])
        self.abs_rel.append(res_dict['abs_rel'])
        self.abs_rel.append(res_dict['sq_rel'])
        self.scalar.append(res_dict['scalar'])

        return s_viz

    def stats_print(self):
        print('----------------------------------------------------------------------------')
        print('RMSE:', np.mean(self.rmse))
        print('ABS_REL:', np.mean(self.abs_rel))
        print('SQ_REL:', np.mean(self.sq_rel))
        print('SCALAR:', np.mean(self.scalar), '       ', np.std(self.scalar))
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
            depth_image_gt = vis_depth(depth_map_gt, max_percent=95, min_percent=0)

            start = time.time()
            depth_map = self.depth_engine.estimate_depth(rgb_image)
            print(time.time() - start)
            if self.enable_filter:
                depth_map = bilateral_filter(rgb_image, depth_map)
            depth_image = vis_depth(depth_map, max_percent=95, min_percent=0)

            if self.save_data:
                self.save_rgbd_data(rgb, depth_image, idx)

            toshow_image = np.hstack((rgb, depth_image, depth_image_gt))

            idx += 1
            Image.fromarray(toshow_image).save('saved_data/' + str(idx).zfill(6) + '.jpg')
            s_viz = self.get_metrics(depth_map, depth_map_gt)
            plt.imshow(toshow_image)
            plt.text(0, 0, s_viz, fontsize=24)
            plt.axis('off')
            plt.title('Nod Depth', fontsize=32)
            plt.show(block=False)

            plt.pause(0.001)
            plt.clf()

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
    args = parser.parse_args()
    import tensorflow as tf
    print(tf.__version__)
    v = App(args)
    v.run()
