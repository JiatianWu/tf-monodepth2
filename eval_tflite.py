import os
import time
import csv
import copy
import numpy as np
import pdb
import pickle
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tflite_runtime.interpreter as tflite
import tensorflow as tf

from conversion.data_io import *
from utils.tools import *
from utils.process_data import *

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

        self.rmse = []
        self.abs_rel = []
        self.sq_rel = []
        self.scalar = []

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

        return np_image

    def EstimateDisp(self, image):
        disp_map = self.Inference(self.TransformImage(image))

        return disp_map

    def get_metrics(self, depth_map, depth_map_gt, min_depth, max_depth):
        res_dict = eval_depth(depth_map, depth_map_gt, min_depth, max_depth)

        self.rmse.append(res_dict['rms'])
        self.abs_rel.append(res_dict['abs_rel'])
        self.sq_rel.append(res_dict['sq_rel'])
        self.scalar.append(res_dict['scalar'])

    def stats_print(self):
        print('----------------------------------------------------------------------------')
        print('RMSE:', np.mean(self.rmse))
        print('ABS_REL:', np.mean(self.abs_rel))
        print('SQ_REL:', np.mean(self.sq_rel))
        print('SCALAR:', np.mean(self.scalar), '       ', np.std(self.scalar))
        print('----------------------------------------------------------------------------')

if __name__ == '__main__':
    model_file = 'saved_model/tflite_640_480_bilinear/saved_model_quant.tflite'
    width = 640 #256
    height = 480 #192
    input_size = (width, height) # (width, height)
    hardware = 'edgetpu'
    min_depth = 0.1
    max_depth = 10.
    depth_estimator = DepthEstimation(model_file=model_file, input_size=input_size, hardware=hardware)

    # input_dir = '/home/nod/datasets/nod/images0_undistorted'
    # output_dir = '/home/nod/datasets/nod/eval'
    # data_source = yield_data_from_datasets_nod(input_dir)
    # data_source = iter(data_source)
    # output_dir = build_output_dir(output_dir=output_dir)

    # step = 0
    # while True:
    #     print('Step: ', step)
    #     image = next(data_source)

    #     if step % 10 != 0:
    #         step += 1
    #         continue
    #     else:
    #         step += 1

    #     image = process_image_eval_tflite_nod(image, width, height, scale=0.5)
    #     disp_map = depth_estimator.EstimateDisp(image)
    #     depth_map = disp_to_depth_np(disp_map, min_depth, max_depth)
    #     depth_image = vis_depth(depth_map)

    #     toshow_image = np.vstack((image, depth_image))
    #     Image.fromarray(toshow_image).save(output_dir + '/' + str(step).zfill(6) + '.jpg')

    # input_dir = '/home/nod/nod/nod/src/apps/nod_depth/saved_data/rgbd_data'
    # output_dir = '/home/nod/datasets/nod/eval'
    # seqs = sorted(os.listdir(input_dir))
    # step = 0
    # for seq in seqs:
    #     data_path = input_dir + '/' + seq
    #     if step % 5 != 0:
    #         step += 1
    #         continue

    #     data = open(data_path, "rb")
    #     rgbd = pickle.load(data)

    #     rgb = rgbd['rgb']
    #     disp_map = depth_estimator.EstimateDisp(rgb)
    #     depth_map = disp_to_depth_np(disp_map, min_depth, max_depth)
    #     depth_image = vis_depth(depth_map)

    #     # data_dict = {'rgb': rgb, 'depth': rgbd['depth'], 'tflite': depth_image}
    #     # data_dict_path = output_dir + '/' + str(idx).zfill(6) + '.pkl'
    #     # f = open(data_dict_path, 'wb')
    #     # pickle.dump(data_dict, f)
    #     # f.close()

    #     step += 1

    #     # toshow_image = np.vstack((rgb, depth_image))
    #     toshow_image = depth_image
    #     Image.fromarray(toshow_image).save(output_dir + '/' + str(step).zfill(6) + '.jpg')

    #     # plt.imshow(depth_image)
    #     # plt.show(block=False)
    #     # plt.pause(0.001)
    #     # plt.clf()

    #-------------------------Eval on NYU dataset-------------------------
    image_source, depth_source = read_process_nyu_data('/home/nod/datasets/nyudepthV2/nyu_depth_v2_labeled.mat')

    for step in range(image_source.shape[0]):
        image = np.transpose(image_source[step], (2, 1, 0))
        depth = np.transpose(depth_source[step], (1, 0))

        image = process_image_eval_tflite_nod(image, width, height, scale=1.0)
        disp_map = depth_estimator.EstimateDisp(image)
        disp_map = np.array(disp_map, dtype=np.float32) * 0.0039065
        depth_map = disp_to_depth_np(disp_map, min_depth, max_depth)

        depth_map_gt = np.array(Image.fromarray(depth).resize((int(depth.shape[1]), int(depth.shape[0]))))

        depth_estimator.get_metrics(depth_map, depth_map_gt, min_depth, max_depth)
        depth_estimator.stats_print()

        data_dict = {'rgb': image, 'depth_pred':depth_map, 'depth_gt':depth_map_gt}
        dict_file_name = '/home/nod/datasets/nyudepthV2/rgbd_gt_tpu_nopp_data/' + str(step).zfill(6) + '.pkl'
        f = open(dict_file_name,"wb")
        pickle.dump(data_dict,f)
        f.close()

        print('Step: ', step)

        # depth_image = vis_depth(depth_map)
        # depth_image_gt = vis_depth(depth_map_gt)
        # toshow_image = np.vstack((image, depth_image, depth_image_gt))
        # Image.fromarray(toshow_image).save('/home/nod/datasets/nyudepthV2/eval_res_images_gray/' + str(step).zfill(6) + '.jpg')

    # image_path = '/home/nod/datasets/nod/debug/tmp.png'
    # image_path = '/home/nod/datasets/nod/images0_undistorted/364838870010.pgm'
    # image = Image.open(image_path).resize(input_size).convert('RGB')
    # image = np.array(image)
    # # image = np.array(ImageOps.equalize(image, mask=None))
    # disp_map = depth_estimator.EstimateDisp(image)
    # disp_map = np.array(disp_map, dtype=np.float32) * 0.0039065
    # depth_map = disp_to_depth_np(disp_map, min_depth, max_depth) * 2.82

    # x_ratio = 320.0/640
    # y_ratio = 240.0/400
    # P_rect = np.eye(3, 3)
    # # Device 91
    # # P_rect[0,0] = 287.94933867261614 * x_ratio
    # # P_rect[0,2] = 324.5096108396658 * x_ratio
    # # P_rect[1,1] = 287.67889279870326 * y_ratio
    # # P_rect[1,2] = 197.44876864233424 * y_ratio

    # # Device 35 undistorted
    # P_rect[0,0] = 238.52264234 * x_ratio
    # P_rect[0,2] = 332.14860019 * x_ratio
    # P_rect[1,1] = 238.15329999 * y_ratio
    # P_rect[1,2] = 195.73207196 * y_ratio

    # pc_save_path = '/home/nod/datasets/nod/debug/tmp_undistorted.ply'
    # pc_save = generate_pointcloud(image, depth_map, P_rect, ply_file=pc_save_path)