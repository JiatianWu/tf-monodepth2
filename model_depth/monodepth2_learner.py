from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from dataloader.supervise_data_loader import DataLoader
from model.net import Net
from utils.tools import *

import matplotlib as mpl
import matplotlib.cm as cm

from tensorflow.python.ops import control_flow_ops


class MonoDepth2Learner(object):
    def __init__(self, **config):
        self.config = config
        self.preprocess = self.config['dataset']['preprocess']
        self.min_depth = np.float(self.config['dataset']['min_depth'])
        self.max_depth = np.float(self.config['dataset']['max_depth'])
        self.root_dir = self.config['model']['root_dir']
        self.pose_type = self.config['model']['pose_type']

        self.resize_bilinear = False

    def preprocess_image(self, image):
        image = (image - 0.45) / 0.225
        return image

    def compute_loss_l1(self, output, label):
        valid_mask = label > 0.001
        diff = tf.abs(output - label)
        diff_valid = tf.boolean_mask(diff, valid_mask)
        loss = tf.reduce_mean(diff_valid)

        return loss

    def build_train(self):
        self.start_learning_rate = np.float(
            self.config['model']['learning_rate'])
        self.total_epoch = np.int(self.config['model']['epoch'])
        self.beta1 = np.float(self.config['model']['beta1'])
        self.continue_ckpt = self.config['model']['continue_ckpt']
        self.torch_res18_ckpt = self.config['model']['torch_res18_ckpt']
        self.summary_freq = self.config['model']['summary_freq']

        loader = DataLoader(trainable=True, **self.config)
        with tf.name_scope('data_loading'):
            src_image_stack, src_depth_stack = loader.load_batch()
            src_image_stack = tf.image.convert_image_dtype(
                src_image_stack, dtype=tf.float32)
            src_depth_stack = tf.image.convert_image_dtype(
                src_depth_stack, dtype=tf.float32) * 65.536

            if self.preprocess:
                src_image_stack_net = self.preprocess_image(
                    src_image_stack)
            else:
                src_image_stack_net = src_image_stack

        with tf.variable_scope('monodepth2_model', reuse=tf.AUTO_REUSE) as scope:
            net_builder = Net(True, **self.config)
            res18_tc, skips_tc = net_builder.build_resnet18(src_image_stack_net)

            if self.resize_bilinear:
                pred_disp = net_builder.build_disp_net_bilinear(res18_tc, skips_tc)
            else:
                pred_disp = net_builder.build_disp_net(res18_tc, skips_tc)[0]

            pred_depth_rawscale = disp_to_depth(pred_disp, self.min_depth, self.max_depth)

        with tf.name_scope('compute_loss'):
            curr_proj_error = tf.abs(pred_depth_rawscale - src_depth_stack)
            total_loss = self.compute_loss_l1(pred_depth_rawscale, src_depth_stack)

        with tf.name_scope('train_op'):
            self.total_step = self.total_epoch * loader.steps_per_epoch
            self.global_step = tf.Variable(
                0, name='global_step', trainable=False)
            learning_rates = [self.start_learning_rate,
                              self.start_learning_rate / 10]
            boundaries = [np.int(self.total_step * 3 / 4)]
            self.learning_rate = tf.train.piecewise_constant(
                self.global_step, boundaries, learning_rates)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(
                    total_loss, global_step=self.global_step)

            self.incr_global_step = tf.assign(
                self.global_step, self.global_step + 1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth_rawscale
        self.pred_disp = pred_disp
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.src_image_stack_all = src_image_stack
        self.src_depth_stack_all = src_depth_stack
        self.pred_depth_stack_all = pred_depth_rawscale
        self.proj_error_stack_all = curr_proj_error

    def collect_summaries(self):
        tf.summary.scalar("total_loss", self.total_loss)
        # tf.summary.image('src_image', self.src_image_stack_all[0])
        # tf.summary.image('depth_color_image', colorize(self.pred_depth_stack_all[0], cmap='plasma'))
        # tf.summary.image('gt_depth_color_image', colorize(self.src_depth_stack_all[0], cmap='plasma'))
        # tf.summary.image('proj_error', self.proj_error_stack_all[0])

    def train(self, ckpt_dir):
        self.build_train()
        init = tf.global_variables_initializer()
        self.collect_summaries()

        # load weights from pytorch resnet 18 model
        if self.torch_res18_ckpt != '':
            assign_ops = load_resnet18_from_file(self.torch_res18_ckpt)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum(
                [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        var_list = [var for var in tf.global_variables()
                    if "moving" in var.name]
        var_list += tf.trainable_variables()
        self.saver = tf.train.Saver(
            var_list + [self.global_step], max_to_keep=10)
        sv = tf.train.Supervisor(
            logdir=ckpt_dir, save_summaries_secs=0, saver=None)
        # print('/n/n/nCollections=====================',tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            # print('Trainable variables: ')
            # for var in var_list:
            #     print(var.name)
            #
            # print('\n\n==========================================')
            # print('Model variables:')
            # for var in tf.model_variables():
            #     print(var.name)
            #
            # print('\n\n==========================================')
            # print('Global variables:')
            # for var in tf.global_variables():
            #     print(var.name)

            print("parameter_count =", sess.run(parameter_count))
            sess.run(init)

            if self.continue_ckpt != '':
                print("Resume training from previous checkpoint: %s" %
                      self.continue_ckpt)
                # ckpt = tf.train.latest_checkpoint('{}/{}'.format(self.root_dir,self.continue_ckpt))
                self.saver.restore(sess, self.continue_ckpt)

            elif self.torch_res18_ckpt != '':
                sess.run(assign_ops)

            start_time = time.time()
            try:
                for step in range(0, self.total_step):
                    fetches = {
                        "train": self.train_op,
                        "global_step": self.global_step,
                        "incr_global_step": self.incr_global_step
                    }

                    if step % self.summary_freq == 0:
                        fetches["loss"] = self.total_loss
                        fetches["summary"] = sv.summary_op
                        fetches["lr"] = self.learning_rate

                    print('Process step: ', step)
                    results = sess.run(fetches)
                    gs = results["global_step"]
                    print('End Process step: ', step)

                    if step % self.summary_freq == 0:
                        sv.summary_writer.add_summary(results["summary"], gs)
                        train_epoch = math.ceil(gs / self.steps_per_epoch)
                        train_step = gs - (train_epoch - 1) * \
                            self.steps_per_epoch
                        print("Epoch: [{}] | [{}/{}] | time: {:.4f} s/it | loss: {:.4f} | lr: {:.5f}".format
                              (train_epoch, train_step, self.steps_per_epoch,
                               (time.time() - start_time) /
                               self.summary_freq,
                               results["loss"], results["lr"]))
                        start_time = time.time()

                    if step != 0 and step % (self.steps_per_epoch * 2) == 0:
                        self.save(sess, ckpt_dir, gs)
            except:
                self.save(sess, ckpt_dir, 'latest')

            self.save(sess, ckpt_dir, 'latest')

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to {}...".format(checkpoint_dir))
        if step == 'latest':
            self.saver.save(sess, os.path.join(
                checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, os.path.join(
                checkpoint_dir, model_name), global_step=step)