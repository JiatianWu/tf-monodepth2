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

def read_nod_datasets(input_dir):
    all_frames = []
    # N = len(glob(input_dir + '/*.pgm'))
    # for n in range(N):
    #     frame_id = str(n).zfill(6)
    #     all_frames.append(input_dir + '/' + frame_id + '.jpg')
    for file in sorted(os.listdir(input_dir)):
        all_frames.append(input_dir + '/' + file)

    return all_frames

def yield_data_from_datasets(input_dir):
    N = len(glob(input_dir + '/*.jpg'))
    image = None
    for n in range(1, N+1, 5):
        frame_id = str(n).zfill(6)
        image_path = input_dir + '/' + frame_id + '.jpg'

        yield cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def get_image(all_frames, id, resize_ratio, crop=False, width=320):
    image_path = all_frames[id]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop:
        image = image[: , width: width*2, :]
    tgt_image_np = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
    tgt_image_np_full = np.zeros((480, 640, 3), dtype=np.uint8)
    tgt_image_np_full[:400, :, :] = tgt_image_np
    tgt_image_np = np.expand_dims(tgt_image_np_full, axis=0)

    return tgt_image_np

def process_image_eval_tflite(image, width, height, resize_ratio_width, resize_ratio_height):
    image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio_width), int(image.shape[0]*resize_ratio_height)))

    tgt_image_np = image[: height, : width, :]
    tgt_depth_np = image[height :, : width, :]

    return tgt_image_np, tgt_depth_np

def disp_to_depth_np(disp, min_depth, max_depth):
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = np.float(min_disp) + np.float(max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return depth

def get_image_depth(results, tgt_image_np, min_depth, max_depth):
    disp_resized_np = np.squeeze(results['disp'])

    colormapped_depth = vis_disparity(disp_resized_np, min_depth, max_depth)
    tgt_image_np = vis_image(tgt_image_np)
    toshow_image = np.vstack((tgt_image_np, colormapped_depth))

    return toshow_image

def vis_image(image):
    if image.ndim > 3:
        image = np.squeeze(image)

    tgt_image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return tgt_image_np

def vis_disparity(disp, min_depth, max_depth):
    disp_resized_vis = disp_to_depth_np(disp, min_depth, max_depth)
    vmax = np.percentile(disp_resized_vis, 95)
    vmin = disp.min()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='viridis')

    colormapped_im = (mapper.to_rgba(disp_resized_vis)[:, :, :3][:,:,::-1] * 255).astype(np.uint8)

    return colormapped_im

def vis_depth(depth_map, cmap):
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    depth_relative = (depth_map - d_min) / (d_max - d_min)

    return 255 * cmap(depth_relative)[:,:,:3]

def merge_image(img_a, img_b, img_c=None, img_d=None):
    res_image = np.vstack((img_a, img_b))

    if img_c is not None:
        res_image = np.vstack((res_image, img_c))

    if img_d is not None:
        res_image = np.vstack((res_image, img_d))

    return res_image