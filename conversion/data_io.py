from __future__ import division
import os
import time
import math
import numpy as np
from glob import glob

import cv2
import matplotlib as mpl
import matplotlib.cm as cm

def build_output_dir(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=False)
    except:
        os.makedirs(output_dir, exist_ok=True)

    return output_dir

def read_datasets(input_dir):
    all_frames = []
    N = len(glob(input_dir + '/*.jpg'))
    for n in range(N):
        frame_id = str(n).zfill(6)
        all_frames.append(input_dir + '/' + frame_id + '.jpg')

    return all_frames

def get_image(all_frames, id):
    width = 320
    image_path = all_frames[id]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tgt_image_np = image[:, width: width*2, :]
    tgt_image_np = np.expand_dims(tgt_image_np, axis=0)

    return tgt_image_np

def disp_to_depth_np(disp, min_depth, max_depth):
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = np.float(min_disp) + np.float(max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return depth

def get_depth(results, tgt_image_np):
    disp_resized_np = np.squeeze(results['disp'])

    disp_resized_vis = disp_to_depth_np(disp_resized_np, 0.1, 100.0)
    vmax = np.percentile(disp_resized_vis, 95)
    vmin = disp_resized_np.min()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='viridis')

    colormapped_im = (mapper.to_rgba(disp_resized_vis)[:, :, :3][:,:,::-1] * 255).astype(np.uint8)

    # disp_rgb_int = ((disp_resized_np - vmin) / (vmax - vmin) * 255.).astype(np.uint8)
    # disp_rgb = cv2.cvtColor( disp_rgb_int, cv2.COLOR_GRAY2RGB)

    tgt_image_np = np.squeeze(tgt_image_np)
    tgt_image_np = cv2.cvtColor(tgt_image_np, cv2.COLOR_BGR2RGB)

    toshow_image = np.vstack((tgt_image_np, colormapped_im))

    return toshow_image