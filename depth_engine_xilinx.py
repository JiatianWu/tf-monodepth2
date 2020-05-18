#Copyright 2020 Nod Labs
from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.decent_q

from utils.tools import *

class DepthEngine(object):
    def __init__(self, **config):
        self.config = config['config']

        self.min_depth = np.float(self.config['dataset']['min_depth'])
        self.max_depth = np.float(self.config['dataset']['max_depth'])
        self.img_width = self.config['dataset']['image_width']
        self.img_height = self.config['dataset']['image_height']
        self.ckpt_dir = self.config['model']['continue_ckpt']
        self.pb_path = self.config['model']['pb_path']
        self.inference_ckpt = self.config['model']['inference_ckpt']

    def preprocess_image(self, image):
        image = (image - 0.45) / 0.225
        return image

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

    def load_model(self):
        if self.inference_ckpt:
            self.build_depth()

            var_list = [var for var in tf.global_variables() if "moving" in var.name]
            var_list += tf.trainable_variables()
            self.saver = tf.train.Saver(var_list, max_to_keep=10)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.saver.restore(self.sess, self.ckpt_dir)

            self.fetches = {'disp': self.pred_disp}
        else:
            self.graph = tf.Graph()

            with tf.io.gfile.GFile(self.pb_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            with self.graph.as_default():
                # Define input tensor
                self.input_tensor = tf.compat.v1.placeholder(tf.float32,
                            shape = [None, self.img_height, self.img_width, 3], name='input')
                tf.import_graph_def(graph_def, {'input:0': self.input_tensor})

            self.graph.finalize()
            self.sess = tf.compat.v1.Session(graph = self.graph)
            self.output_tensor = self.graph.get_tensor_by_name('import/output:0')

    def estimate_depth(self, image):
        if image.ndim < 4:
            image = np.expand_dims(image, axis=0)

        if self.inference_ckpt:
            results = self.sess.run(self.fetches,feed_dict={self.tgt_image_uint8:image})
            disp = np.squeeze(results['disp'])
        else:
            feed_dict = {self.input_tensor: image}
            result = self.sess.run(self.output_tensor, feed_dict=feed_dict)
            disp = np.squeeze(result)

        return disp_to_depth_np(disp, self.min_depth, self.max_depth)