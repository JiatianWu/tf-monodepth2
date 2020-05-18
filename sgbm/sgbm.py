#####################################################################

# Example : stereo vision from 2 connected cameras using Semi-Global
# Block Matching. For usage: python3 ./stereo_sgbm.py -h

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015-18 Engineering & Computer Science, Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

# Acknowledgements:

# http://opencv-python-tutroals.readthedocs.org/en/latest/ \
# py_tutorials/py_calib3d/py_table_of_contents_calib3d/py_table_of_contents_calib3d.html

# http://docs.ros.org/electric/api/cob_camera_calibration/html/calibrator_8py_source.html
# OpenCV 3.0 example - stereo_match.py

# Andy Pound, Durham University, 2016 - calibration save/load approach

#####################################################################

# TODO:

# add sliders for some stereo parameters


#####################################################################

import cv2
import sys
import numpy as np
import os
import argparse
import time
from PIL import Image

sys.path.append(os.path.abspath(os.path.join('..', 'utils')))

from tools import *

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

def stereo_global_matching(image_lf, image_rt, baseline, fx):
    # remember to convert to grayscale (as the disparity matching works on grayscale)
    gray_lf = cv2.cvtColor(image_lf, cv2.COLOR_BGR2GRAY)
    gray_rt = cv2.cvtColor(image_rt, cv2.COLOR_BGR2GRAY)

    # compute disparity image from undistorted and rectified versions
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    disparity = stereoProcessor.compute(gray_lf, gray_rt)
    cv2.filterSpeckles(disparity, 0, 40, max_disparity)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available
    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)

    depth = disp_to_depth(disparity, baseline, fx)

    return depth

def disp_to_depth(disp, baseline, fx):
    depth = baseline * fx / (np.array(disp, np.float) + 0.01)
    depth[depth < 0.1] = 0.1
    depth[depth > 58.45] = 58.45
    return depth

if __name__ == "__main__":
    import pickle

    path = '/home/nod/datasets/weanhall/rectified'
    image_left_path_list = []
    image_right_path_list = []
    image_list = sorted(os.listdir(path))
    for image in image_list:
        if 'left' in image:
            image_left_path_list.append(path + '/' + image)
        elif 'right' in image:
            image_right_path_list.append(path + '/' + image)

    batch_size = len(image_left_path_list)
    for idx in range(batch_size):
        print('Processing ', idx)
        image_left = cv2.imread(image_left_path_list[idx])
        image_right = cv2.imread(image_right_path_list[idx])

        depth = stereo_global_matching(image_left, image_right, 0.120006, 487.109)
        data_dict = {'depth': depth}
        data_file_name = '/home/nod/datasets/weanhall/eval_sgm_data/' + str(idx).zfill(6) + '.pkl'
        f = open(data_file_name,"wb")
        pickle.dump(data_dict,f)
        f.close()
        # depth_image = vis_depth(depth)
        # toshow_image = np.hstack((image_left, image_right, depth_image))
        # Image.fromarray(toshow_image).save('/home/nod/datasets/weanhall/eval_sgm/' + str(idx).zfill(6) + '.jpg')
