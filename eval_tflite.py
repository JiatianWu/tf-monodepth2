import os
import time
import csv
import copy
import numpy as np
import pdb
import pickle
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import tflite_runtime.interpreter as tflite
import tensorflow as tf

from conversion.data_io import *

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

def SetInput(interpreter, image):
    tensor = input_tensor(interpreter)
    tensor[:, :] = image

def output_tensor(interpreter, i):
  """Returns output tensor view."""
  tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
  return np.squeeze(tensor)

def GetOutput(interpreter):
    tensor = output_tensor(interpreter, 0)
    data = copy.deepcopy(np.squeeze(tensor))
    return data

class DepthEstimation():
    def __init__(self, model_file, input_size, hardware):
        self.model_file = model_file
        self.interpreter = self.MakeInterpreter(hardware=hardware)
        self.interpreter.allocate_tensors()
        self.input_size = input_size

        self.cmap = plt.cm.viridis

    def MakeInterpreter(self, hardware='edgetpu'):
        model_file, *device = self.model_file.split('@')

        if hardware == 'cpu':
            return tflite.Interpreter(model_path=model_file)
        elif hardware == 'edgetpu':
            return tflite.Interpreter(model_path=model_file,
                                      experimental_delegates=[
                                            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                                        {'device': device[0]} if device else {})])
        else:
            print("Hardware Not support!")

    def Inference(self, image):
        start = time.monotonic()
        SetInput(self.interpreter, image)

        self.interpreter.invoke()

        output_data = GetOutput(self.interpreter)
        inference_time = time.monotonic() - start
        print('Inference takes %.1fms' % (inference_time * 1000))

        return output_data

    def TransformImage(self, image):
        np_image = np.array(Image.fromarray(image).resize(self.input_size))
        if np_image.ndim < 3:
            np_image = np.repeat(np_image[:, :, np.newaxis], 3, axis=2)
        np_image = np_image.astype(np.float) / 255.0

        return np_image

    def EstimateDisp(self, image):
        disp_map = self.Inference(self.TransformImage(image))

        return disp_map

if __name__ == '__main__':
    model_file = 'saved_model/tflite_640_480/saved_model_quant.tflite'
    width = 640 #256
    height = 480 #192
    input_size = (width, height) # (width, height)
    hardware = 'cpu'
    min_depth = 0.1
    max_depth = 10.
    depth_estimator = DepthEstimation(model_file=model_file, input_size=input_size, hardware=hardware)

    # input_dir = '/home/jiatian/dataset/tmp_test/tmp_tello_2'
    # output_dir = '/home/jiatian/dataset/tmp_test/tmp_tello_tflite_maxdepth10_nod_320_240_0215'
    # data_source = yield_data_from_datasets(input_dir)
    # data_source = iter(data_source)
    # output_dir = build_output_dir(output_dir=output_dir)

    # step = 0
    # while True:
    #     step += 1
    #     print('Step: ', step)
    #     image = next(data_source)

    #     resize_ratio_width = np.float(width/320.) / 3
    #     resize_ratio_height = np.float(height/240.) / 3
    #     image, gt_depth = process_image_eval_tflite(image, width, height, resize_ratio_width, resize_ratio_height)
    #     disp_map = depth_estimator.EstimateDisp(image)

    #     eval_depth = vis_disparity(disp_map, min_depth, max_depth)
    #     eval_depth = Image.fromarray(eval_depth).resize((width, height))

    #     toshow_image = merge_image(vis_image(image), vis_image(gt_depth), eval_depth)
    #     cv2.imwrite(output_dir + '/' + str(step).zfill(6) + '.jpg', toshow_image)

    input_dir = '/home/nod/nod/nod/src/apps/nod_depth/saved_data/rgbd_data'
    output_dir = '/home/nod/nod/nod/src/apps/nod_depth/saved_data/rgbd_tflite_data'
    seqs = sorted(os.listdir(input_dir))
    idx = 0
    for seq in seqs:
        data_path = input_dir + '/' + seq
        if idx % 5 != 0:
            idx += 1
            continue

        data = open(data_path, "rb")
        rgbd = pickle.load(data)

        rgb = rgbd['rgb']
        disp_map = depth_estimator.EstimateDisp(rgb)
        depth_map = disp_to_depth_np(disp_map, min_depth, max_depth)
        depth_image = vis_depth(depth_map)

        data_dict = {'rgb': rgb, 'depth': rgbd['depth'], 'tflite': depth_image}
        data_dict_path = output_dir + '/' + str(idx).zfill(6) + '.pkl'
        f = open(data_dict_path, 'wb')
        pickle.dump(data_dict, f)
        f.close()

        idx += 1

        plt.imshow(depth_image)
        plt.show(block=False)
        plt.pause(0.001)
        plt.clf()