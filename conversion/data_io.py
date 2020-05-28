from __future__ import division
import os
import sys
import time
import math
import numpy as np
from glob import glob

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image

sys.path.append(os.path.abspath(os.path.join('..', 'utils')))
from utils.bilateral_filter import bilateral_filter

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
    for file in sorted(os.listdir(input_dir)):
        if 'jpg' in file or 'png' in file or 'pgm' in file: 
            all_frames.append(input_dir + '/' + file)

    return all_frames

def yield_data_from_datasets(input_dir):
    N = len(glob(input_dir + '/*.jpg'))
    image = None
    for n in range(1, N+1, 10):
        frame_id = str(n).zfill(6)
        image_path = input_dir + '/' + frame_id + '.jpg'

        yield cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def yield_data_from_datasets_nod(input_dir):
    dirlist = sorted(os.listdir(input_dir))
    for path in dirlist:
        image_path = input_dir + '/' + path

        yield cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def get_image(all_frames, id, resize_ratio, crop=False, mask_height=False, output_src_image=False):
    image_path = all_frames[id]
    image = np.array(Image.open(image_path))
    if crop:
        width_orig = image.shape[1]
        width = int(width_orig / 3)
        tgt_image = image[: , width: width*2, :]
        if output_src_image:
            src_image = image[: , : width, :]

    tgt_image_np = cv2.resize(tgt_image, dsize=(int(tgt_image.shape[1]*resize_ratio), int(tgt_image.shape[0]*resize_ratio)))
    # tgt_image_np = cv2.resize(image, dsize=(640, 192))
    tgt_image_np_full = np.zeros((tgt_image_np.shape[0], tgt_image_np.shape[1], 3), dtype=np.uint8)
    if mask_height:
        tgt_image_np_full[:400, :, :] = tgt_image_np
    else:
        tgt_image_np_full[:, :, :] = tgt_image_np
    tgt_image_np = np.expand_dims(tgt_image_np_full, axis=0)

    if output_src_image:
        src_image_np = cv2.resize(src_image, dsize=(int(src_image.shape[1]*resize_ratio), int(src_image.shape[0]*resize_ratio)))
        src_image_np_full = np.zeros((src_image_np.shape[0], src_image_np.shape[1], 3), dtype=np.uint8)
        if mask_height:
            src_image_np_full[:400, :, :] = src_image_np
        else:
            src_image_np_full[:, :, :] = src_image_np
        src_image_np = np.expand_dims(src_image_np_full, axis=0)

        return tgt_image_np, src_image_np

    return tgt_image_np

def get_nod_image(all_frames, id, resize_ratio, crop=False, mask_height=False, output_src_image=False):
    image_path = all_frames[id]
    image = np.array(Image.open(image_path).convert("RGB"))
    if crop:
        width_orig = image.shape[1]
        width = int(width_orig / 3)
        tgt_image = image[: , width: width*2, :]
        if output_src_image:
            src_image = image[: , : width, :]
    else:
        tgt_image = image

    tgt_image_np = cv2.resize(tgt_image, dsize=(int(tgt_image.shape[1]*resize_ratio), int(tgt_image.shape[0]*resize_ratio)))
    # tgt_image_np = cv2.resize(image, dsize=(640, 192))
    tgt_image_np_full = np.zeros((480, 640, 3), dtype=np.uint8)
    if mask_height:
        tgt_image_np_full[:400, :, :] = tgt_image_np
    else:
        tgt_image_np_full[:, :, :] = tgt_image_np
    tgt_image_np = np.expand_dims(tgt_image_np_full, axis=0)

    if output_src_image:
        src_image_np = np.expand_dims(image, axis=0)

        return tgt_image_np, src_image_np

    return tgt_image_np

def process_image_eval_tflite(image, width, height, resize_ratio_width, resize_ratio_height):
    image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio_width), int(image.shape[0]*resize_ratio_height)))

    tgt_image_np = image[: height, : width, :]
    tgt_depth_np = image[height :, : width, :]

    return tgt_image_np, tgt_depth_np

def process_image_eval_tflite(image, width, height, resize_ratio_width, resize_ratio_height):
    image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio_width), int(image.shape[0]*resize_ratio_height)))

    tgt_image_np = image[: height, : width, :]
    tgt_depth_np = image[height :, : width, :]

    return tgt_image_np, tgt_depth_np

def rgb2gray(R, G, B):
    Y = 0.2125 * R + 0.7154 * G + 0.0721 *B

    return Y

def process_image_eval_tflite_nod(image, width, height, scale=1.0, use_gray=False):
    image = np.array(Image.fromarray(image).resize((int(image.shape[1]*scale), int(image.shape[0]*scale))))
    if use_gray:
        image_gray = rgb2gray(image[:,:,0], image[:,:,1], image[:,:,2])
        image_rgb = np.array(Image.fromarray(image_gray).convert('RGB'))
        image = image_rgb
    np_image = np.zeros((height, width, 3), dtype=np.uint8)
    np_image[:image.shape[0], :image.shape[1], :] = image

    return np_image

def disp_to_depth_np(disp, min_depth, max_depth):
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = np.float(min_disp) + np.float(max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return depth

def get_image_depth(results, tgt_image_np, min_depth, max_depth, use_bf=True, mask_height=False):
    disp_resized_np = np.squeeze(results['disp'])

    if mask_height:
        tgt_image_np_valid = tgt_image_np[0, :400, :, :]
        disp_resized_np_valid = disp_resized_np[:400, :]
    else:
        tgt_image_np_valid = tgt_image_np[0, :, :, :]
        disp_resized_np_valid = disp_resized_np[:, :]

    depth_map = disp_to_depth_np(disp_resized_np_valid, min_depth, max_depth)
    if use_bf:
        start = time.time()
        depth_map = bilateral_filter(tgt_image_np_valid, depth_map)
        print('Bilateral filter takes: ', time.time() - start)

    colormapped_depth = vis_depth(depth_map, color_mode='viridis')
    tgt_image_np = vis_image(tgt_image_np)
    toshow_image = np.vstack((tgt_image_np, colormapped_depth))

    return toshow_image

def get_nod_image_depth(results, tgt_image_np, min_depth, max_depth, use_bf=True, mask_height=False):
    disp_resized_np = np.squeeze(results['disp'])

    if mask_height:
        tgt_image_np_valid = tgt_image_np[0, :400, :, :]
        disp_resized_np_valid = disp_resized_np[:400, :]
    else:
        tgt_image_np_valid = tgt_image_np[0, :, :, :]
        disp_resized_np_valid = disp_resized_np[:, :]

    depth_map = disp_to_depth_np(disp_resized_np_valid, min_depth, max_depth)
    if use_bf:
        start = time.time()
        depth_map = bilateral_filter(tgt_image_np_valid, depth_map)
        print('Bilateral filter takes: ', time.time() - start)

    colormapped_depth = vis_depth(depth_map, color_mode='viridis')
    tgt_image_np = vis_image(tgt_image_np_valid)
    toshow_image = np.hstack((tgt_image_np, colormapped_depth))

    return toshow_image

def vis_image(image):
    if image.ndim > 3:
        image = np.squeeze(image)

    return image

def vis_depth(depth_map, color_mode='viridis'):
    vmax = np.percentile(depth_map, 95)
    vmin = depth_map.min()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap=color_mode)

    colormapped_im = (mapper.to_rgba(depth_map)[:, :, :3][:,:,::-1] * 255).astype(np.uint8)

    return colormapped_im[:, :, ::-1]

def merge_image(img_a, img_b, img_c=None, img_d=None):
    res_image = np.vstack((img_a, img_b))

    if img_c is not None:
        res_image = np.vstack((res_image, img_c))

    if img_d is not None:
        res_image = np.vstack((res_image, img_d))

    return res_image