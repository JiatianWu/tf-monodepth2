from __future__ import division
import os
import time
import math
import pickle
import numpy as np
from glob import glob
import tensorflow as tf
from utils.tools import *

import cv2
import matplotlib as mpl
import matplotlib.cm as cm

from conversion.data_io import *
sys.path.append(os.path.abspath(os.path.join('..', 'utils')))

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
        num_calibration_steps = 600
        for _ in range(num_calibration_steps):
            input = np.random.random_sample((1, self.img_height, self.img_width, 3))
            yield [np.array(input, dtype='float32')]

    # def generate_datasets(self):
    #     num_calibration_steps = 600
    #     for _ in range(num_calibration_steps):
    #         input = np.random.randint(256, size=(1, self.img_height, self.img_width, 3))
    #         yield [np.array(input, dtype=np.uint8)]

    def convert_savedModel_quant_tflite(self, savedModel_dir):
        converter = tf.lite.TFLiteConverter.from_saved_model(savedModel_dir)
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.generate_datasets
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_quant_model = converter.convert()
        save_tflite_path = savedModel_dir + '/saved_model_quant.tflite'
        open(save_tflite_path, "wb").write(tflite_quant_model)

    def build_orig_depth(self):
        from model.net import Net

        self.tgt_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input')
        with tf.name_scope('data_loading'):
            tgt_image_float = tf.image.convert_image_dtype(self.tgt_image_uint8, dtype=tf.float32)
        self.tgt_image_float = tf.identity(tgt_image_float, name='input_float')
        tgt_image_net = self.preprocess_image(self.tgt_image_float)

        # self.tgt_image_float = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, 3], name='input')
        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(tgt_image_net)
            pred_disp = net_builder.build_disp_net(res18_tc, skips_tc)[0]

        self.pred_disp = tf.identity(pred_disp, name='output')

    def build_new_depth(self):
        from model_new.net import Net

        self.tgt_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input')
        with tf.name_scope('data_loading'):
            tgt_image_float = tf.image.convert_image_dtype(self.tgt_image_uint8, dtype=tf.float32)
        self.tgt_image_float = tf.identity(tgt_image_float, name='input_float')
        tgt_image_net = self.preprocess_image(self.tgt_image_float)

        # self.tgt_image_float = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, 3], name='input')
        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(tgt_image_net)
            pred_disp = net_builder.build_disp_net(res18_tc, skips_tc)[0]

        self.pred_disp = tf.identity(pred_disp, name='output')

    def build_depth(self):
        from model_new.net import Net

        self.tgt_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input_uint8')
        with tf.name_scope('data_loading'):
            tgt_image_float = tf.image.convert_image_dtype(self.tgt_image_uint8, dtype=tf.float32)
        self.tgt_image_float = tf.identity(tgt_image_float, name='input')
        tgt_image_net = self.preprocess_image(self.tgt_image_float)

        # self.tgt_image_float = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, 3], name='input')
        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(tgt_image_net)
            pred_disp = net_builder.build_disp_net_bilinear(res18_tc, skips_tc)[0]

        self.pred_disp = tf.identity(pred_disp, name='output')

    def build_xilinx_depth(self):
        from model_xilinx.net import Net

        self.tgt_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input')
        with tf.name_scope('data_loading'):
            tgt_image_float = tf.image.convert_image_dtype(self.tgt_image_uint8, dtype=tf.float32)
        self.tgt_image_float = tf.identity(tgt_image_float, name='input_float')
        tgt_image_net = self.preprocess_image(self.tgt_image_float)

        # self.tgt_image_float = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, 3], name='input')
        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(tgt_image_net)
            pred_disp = net_builder.build_disp_net(res18_tc, skips_tc)[0]

        self.pred_disp = tf.identity(pred_disp, name='output')

    def build_xilinx_depth_postprocess(self):
        from model_xilinx.net import Net

        # self.tgt_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input_uint8')
        # with tf.name_scope('data_loading'):
        #     self.tgt_image_float = tf.image.convert_image_dtype(self.tgt_image_uint8, dtype=tf.float32)
        # tgt_image_net = self.preprocess_image(self.tgt_image_float)
        self.tgt_image_float_postprocess = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, 3], name='input')

        # self.tgt_image_float = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, 3], name='input')
        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(self.tgt_image_float_postprocess)
            pred_disp = net_builder.build_disp_net(res18_tc, skips_tc)[0]

        self.pred_disp = tf.identity(pred_disp, name='output')

    def build_xilinx_depth_postprocess_nosigmoid(self):
        from model_xilinx.net import Net

        # self.tgt_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input_uint8')
        # with tf.name_scope('data_loading'):
        #     self.tgt_image_float = tf.image.convert_image_dtype(self.tgt_image_uint8, dtype=tf.float32)
        # tgt_image_net = self.preprocess_image(self.tgt_image_float)
        self.tgt_image_float_postprocess = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, 3], name='input')

        # self.tgt_image_float = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, 3], name='input')
        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(self.tgt_image_float_postprocess)
            pred_disp = net_builder.build_disp_net_nosigmoid(res18_tc, skips_tc)[0]

        self.pred_disp = tf.identity(pred_disp, name='output')

    def build_default_depth(self):
        from model_new.net import Net

        self.tgt_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input_uint8')
        with tf.name_scope('data_loading'):
            tgt_image_float = tf.image.convert_image_dtype(self.tgt_image_uint8, dtype=tf.float32)
        self.tgt_image_float = tf.identity(tgt_image_float, name='input')
        tgt_image_net = self.preprocess_image(self.tgt_image_float)

        # self.tgt_image_float = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, 3], name='input')
        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(tgt_image_net)
            pred_disp = net_builder.build_disp_net_eval(res18_tc, skips_tc)[0]

        self.pred_disp = tf.identity(pred_disp, name='output')

    def build_depth_pose(self):
        from model.net import Net

        self.tgt_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input_tgt_uint8')
        self.src_image_uint8 = tf.placeholder(tf.uint8, [1, self.img_height, self.img_width, 3], name='input_src_uint8')
        with tf.name_scope('data_loading'):
            tgt_image = tf.image.convert_image_dtype(self.tgt_image_uint8, dtype=tf.float32)
            src_image = tf.image.convert_image_dtype(self.src_image_uint8, dtype=tf.float32)
        self.tgt_image_float = tf.identity(tgt_image, name='input_tgt_float')
        self.src_image_float = tf.identity(src_image, name='input_src_float')
        tgt_image_net = self.preprocess_image(self.tgt_image_float)
        src_image_net = self.preprocess_image(self.src_image_float)

        # self.tgt_image_float = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, 3], name='input')
        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(False, **self.config)

            res18_tc, skips_tc = net_builder.build_resnet18(tgt_image_net)
            pred_disp = net_builder.build_disp_net(res18_tc, skips_tc)[0]

            res18_ctp, _ = net_builder.build_resnet18(
                tf.concat([src_image_net, tgt_image_net], axis=3),
                prefix='pose_'
            )

            pred_pose_ctp = net_builder.build_pose_net2(res18_ctp)

        self.pred_disp = tf.identity(pred_disp, name='output_disp')
        self.pred_pose = tf.identity(pred_pose_ctp, name='output_pose')

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
        input_image_uint8 = np.random.randint(256, size=(1, self.img_height, self.img_width, 3), dtype=np.uint8)
        with tf.Session() as sess:
            self.saver.restore(sess, ckpt_dir)
            feed_dict = {self.tgt_image_float: input_image}
            result = sess.run(self.pred_disp, feed_dict=feed_dict)

            tf.saved_model.simple_save(
                sess,
                savedModel_dir,
                inputs={'input': self.tgt_image_float},
                outputs={'output': self.pred_disp} )

        if save_tflite:
            self.convert_savedModel_quant_tflite(savedModel_dir)

    def save_orig_pb(self, ckpt_dir, pb_path):
        from tensorflow.python.framework import graph_util

        self.build_orig_depth()

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

    def save_pb(self, ckpt_dir, pb_path):
        from tensorflow.python.framework import graph_util

        self.build_new_depth()

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

    def save_xilinx_pb(self, ckpt_dir, pb_path):
        from tensorflow.python.framework import graph_util

        self.build_xilinx_depth()

        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        print('##########################Model Variables###################')
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

    def save_xilinx_pb_postprocess(self, ckpt_dir, pb_path):
        from tensorflow.python.framework import graph_util

        self.build_xilinx_depth_postprocess()

        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        print('##########################Model Variables###################')
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

    def save_xilinx_pb_postprocess_nosigmoid(self, ckpt_dir, pb_path):
        from tensorflow.python.framework import graph_util

        self.build_xilinx_depth_postprocess_nosigmoid()

        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        print('##########################Model Variables###################')
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

    def build_restore_model(self, ckpt_dir):
        self.build_depth_pose()
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver.restore(self.sess, ckpt_dir)

    def build_default_depth_model(self, ckpt_dir):
        self.build_default_depth()
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver.restore(self.sess, ckpt_dir)

    def build_new_depth_model(self, ckpt_dir):
        self.build_new_depth()
        var_list = [var for var in tf.global_variables() if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver.restore(self.sess, ckpt_dir)

    def test_dir_depth_pose(self, input_dir, output_dir):
        all_frames = read_nod_datasets(input_dir=input_dir)
        output_dir = build_output_dir(output_dir=output_dir)

        num_frames = len(all_frames)
        poses_log = {}
        for step in range(num_frames):

            fetches = {
                'disp': self.pred_disp,
                'pose': self.pred_pose
            }

            try:
                tgt_image_np, src_image_np = get_image(all_frames, step, resize_ratio=1.0, crop=True, output_src_image=True)
            except:
                print('Failed to get image!')
                continue

            start = time.time()
            results = self.sess.run(fetches,feed_dict={self.tgt_image_uint8:tgt_image_np, self.src_image_uint8:src_image_np})
            print('Inference takes: ', time.time() - start)

            poses_log[step] = np.squeeze(results['pose'])
            with open(output_dir + '/poses_log.pickle', 'wb') as handle:
                pickle.dump(poses_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # toshow_image = get_image_depth(results=results, tgt_image_np=tgt_image_np,
            #                                 min_depth=self.min_depth, max_depth=self.max_depth, use_bf=True)

            # # toshow_image = cv2.resize(toshow_image, (toshow_image.shape[1]*3, toshow_image.shape[0]*3))
            # Image.fromarray(toshow_image).save(output_dir + '/' + str(step).zfill(6) + '.jpg')

    def test_dir(self, input_dir, output_dir):
        all_frames = read_nod_datasets(input_dir=input_dir)
        output_dir = build_output_dir(output_dir=output_dir)

        num_frames = len(all_frames)
        for step in range(num_frames):

            fetches = {
                'disp': self.pred_disp,
            }

            try:
                tgt_image_np = get_image(all_frames, step, resize_ratio=640.0/600, crop=False)
            except:
                print('Failed to get image!')
                continue

            start = time.time()
            results = self.sess.run(fetches,feed_dict={self.tgt_image_uint8:tgt_image_np})
            print('Inference takes: ', time.time() - start)

            toshow_image = get_image_depth(results=results, tgt_image_np=tgt_image_np,
                                            min_depth=self.min_depth, max_depth=self.max_depth, use_bf=False)

            # toshow_image = cv2.resize(toshow_image, (toshow_image.shape[1]*3, toshow_image.shape[0]*3))
            Image.fromarray(toshow_image).save(output_dir + '/' + str(step).zfill(6) + '.jpg')

    def test_nod_dir(self, input_dir, output_dir):
        all_frames = read_nod_datasets(input_dir=input_dir)
        output_dir = build_output_dir(output_dir=output_dir)

        num_frames = len(all_frames)
        for step in range(num_frames):

            fetches = {
                'disp': self.pred_disp,
            }

            tgt_image_np = get_nod_image(all_frames, step, resize_ratio=1.0, crop=False, mask_height=True)

            start = time.time()
            results = self.sess.run(fetches,feed_dict={self.tgt_image_uint8:tgt_image_np})
            print('Inference takes: ', time.time() - start)

            toshow_image = get_nod_image_depth(results=results, tgt_image_np=tgt_image_np,
                                            min_depth=self.min_depth, max_depth=self.max_depth, use_bf=True, mask_height=True)

            # toshow_image = cv2.resize(toshow_image, (toshow_image.shape[1]*3, toshow_image.shape[0]*3))
            Image.fromarray(toshow_image).save(output_dir + '/' + str(step).zfill(6) + '.jpg')