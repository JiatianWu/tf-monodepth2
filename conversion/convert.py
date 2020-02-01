from __future__ import division
import os
import time
import math
import numpy as np
from glob import glob
import tensorflow as tf
from utils.tools import *

import cv2
import matplotlib as mpl
import matplotlib.cm as cm

from conversion.data_io import *

class SaveModel(object):
    def __init__(self, **config):
        self.config = config['config']
        self.min_depth = np.float(self.config['dataset']['min_depth'])
        self.max_depth = np.float(self.config['dataset']['max_depth'])
        self.root_dir = self.config['model']['root_dir']
        self.pose_type = self.config['model']['pose_type']

        self.img_width = self.config['dataset']['image_width']
        self.img_height = self.config['dataset']['image_height']
        self.num_scales = self.config['model']['num_scales']
        self.num_source = self.config['model']['num_source']

    def preprocess_image(self, image):
        image = (image - 0.45) / 0.225
        return image

    def generate_datasets(self):
        num_calibration_steps = 10
        for _ in range(num_calibration_steps):
            input = np.random.random_sample((1, self.img_height, self.img_width, 3))
            yield [np.array(input, dtype='float32')]

    def convert_savedModel_quant_tflite(self, savedModel_dir):
        converter = tf.lite.TFLiteConverter.from_saved_model(savedModel_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.generate_datasets
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        tflite_quant_model = converter.convert()
        save_tflite_path = savedModel_dir + '/saved_model_quant.tflite'
        open(save_tflite_path, "wb").write(tflite_quant_model)

    def build_depth(self):
        from model.net import Net

        self.tgt_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input')
        with tf.name_scope('data_loading'):
            self.tgt_image = tf.image.convert_image_dtype(self.tgt_image_uint8, dtype=tf.float32)
            tgt_image_net = self.preprocess_image(self.tgt_image)

        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(tgt_image_net)
            pred_disp = net_builder.build_disp_net_eval(res18_tc, skips_tc)

        self.pred_disp = tf.identity(pred_disp, name='output')

    def save_savedModel(self, ckpt_dir, savedModel_dir, save_tflite=False):
        if save_tflite and len(os.listdir(savedModel_dir) ) != 0:
            self.convert_savedModel_quant_tflite(savedModel_dir)
            return

        tf.reset_default_graph()
        self.build_depth()

        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        for var in tf.model_variables():
            print(var)

        input_image = np.random.random_sample(size=(1, self.img_height, self.img_width, 3))
        with tf.Session() as sess:
            self.saver.restore(sess, ckpt_dir)
            feed_dict = {self.tgt_image: input_image}
            result = sess.run(self.pred_disp, feed_dict=feed_dict)

            tf.saved_model.simple_save(
                sess,
                savedModel_dir,
                inputs={'input': self.tgt_image},
                outputs={'output': self.pred_disp} )

        if save_tflite:
            self.convert_savedModel_quant_tflite(savedModel_dir)

    def save_pb(self, ckpt_dir, pb_path):
        from tensorflow.python.framework import graph_util

        self.build_depth()

        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        for var in tf.model_variables():
            print(var)

        if ckpt_dir == '':
            print('No pretrained model provided, exit. ')
            raise ValueError

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        input_node_names = ['input']
        output_node_names = ['output']
        with tf.Session(config=config) as sess:
            print("load trained model")
            self.saver.restore(sess, ckpt_dir)
            graph = tf.get_default_graph()
            input_graph_def = graph.as_graph_def()

            output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)
            with tf.gfile.GFile(pb_path, 'wb') as f:
                f.write(output_graph_def.SerializeToString())

    def test_video(self, ckpt_dir, input_dir, output_dir):
        self.build_depth()
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        all_frames = read_datasets(input_dir=input_dir)
        output_dir = build_output_dir(output_dir=output_dir)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        num_frames = len(all_frames)
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, ckpt_dir)

            for step in range(num_frames):

                fetches = {
                    'depth': self.pred_depth,
                    'disp': self.pred_disp
                }

                try:
                    tgt_image_np = get_image(all_frames, step)
                except:
                    continue

                start = time.time()
                results = sess.run(fetches,feed_dict={self.tgt_image_uint8:tgt_image_np})
                print('Inference takes: ', time.time() - start)

                toshow_image = get_depth(results=results, tgt_image_np=tgt_image_np)
                print(toshow_image.shape)

                toshow_image = cv2.resize(toshow_image, (toshow_image.shape[1]*3, toshow_image.shape[0]*3))
                cv2.imwrite(output_dir + '/' + str(step).zfill(6) + '.jpg', toshow_image)