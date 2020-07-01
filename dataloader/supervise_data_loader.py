from __future__ import division
import os
import random
import numpy as np
import tensorflow as tf


class DataLoader(object):
    def __init__(self, trainable=True, **config):
        self.config = config
        self.dataset_dir = self.config['dataset']['root_dir']
        self.batch_size = np.int(self.config['model']['batch_size'])
        self.img_height = np.int(self.config['dataset']['image_height'])
        self.img_width = np.int(self.config['dataset']['image_width'])

        self.trainable = trainable

    def load_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(
            self.dataset_dir, 'train' if self.trainable else 'val')
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'], seed=seed, shuffle=True if self.trainable else False)
        depth_paths_queue = tf.train.string_input_producer(
            file_list['depth_file_list'], seed=seed, shuffle=True if self.trainable else False)
        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image = tf.image.decode_jpeg(image_contents)
        # [H, W, 3] and [H, W, 3 * num_source]
        src_image = self.unpack_image(
            image, self.img_height, self.img_width, 3)

        # Load depths
        depth_reader = tf.WholeFileReader()
        _, depth_contents = depth_reader.read(depth_paths_queue)
        depth = tf.io.decode_png(
            depth_contents, channels=1, dtype=tf.dtypes.uint16)
        # [H, W, 3] and [H, W, 3 * num_source]
        src_depth = self.unpack_image(
            depth, self.img_height, self.img_width, 1)

        # Form training batches
        src_image_stack, src_depth_stack = \
            tf.train.batch([src_image, src_depth], batch_size=self.batch_size)

        # Data augmentation
        image_all = [src_image_stack, src_depth_stack]
        image_all_aug = self.data_augmentation(image_all)

        src_image_stack_aug = image_all_aug[0]
        src_depth_stack_aug = image_all_aug[1]

        return src_image_stack_aug, src_depth_stack_aug

    # edit at 05/26 by Frank
    # add random brightness, contrast, saturation and hue to all source image
    # [H, W, (num_source + 1) * 3]
    def data_augmentation(self, im_all):
        def random_flip(im):
            def flip_one(sim):
                do_flip = tf.random_uniform([], 0, 1)
                return tf.cond(do_flip > 0.5, lambda: [tf.image.flip_left_right(sim[0]), tf.image.flip_left_right(sim[1])], lambda: sim)

            im = tf.map_fn(lambda sim: flip_one(sim), im)
            return im

        def augment_image_properties(im):
            # random brightness
            brightness_seed = random.randint(0, 2**31 - 1)
            im = tf.image.random_brightness(im, 0.2, brightness_seed)

            contrast_seed = random.randint(0, 2 ** 31 - 1)
            im = tf.image.random_contrast(im, 0.8, 1.2, contrast_seed)

            num_img = np.int(im.get_shape().as_list()[-1] // 3)

            # saturation_seed = random.randint(0, 2**31 - 1)
            saturation_im_list = []
            # tf.random_ops.random_uniform([], 0.8, 1.2, seed=saturation_seed)
            saturation_factor = random.uniform(0.8, 1.2)
            for i in range(num_img):
                saturation_im_list.append(tf.image.adjust_saturation(
                    im[:, :, 3*i: 3*(i+1)], saturation_factor))
                # tf.image.random_saturation(im[:,:, 3*i: 3*(i+1)], 0.8, 1.2, seed=saturation_seed))
            im = tf.concat(saturation_im_list, axis=2)

            #hue_seed = random.randint(0, 2 ** 31 - 1)
            hue_im_list = []
            # tf.random_ops.random_uniform([], -0.1, 0.1, seed=hue_seed)
            hue_delta = random.uniform(-0.1, 0.1)
            for i in range(num_img):
                hue_im_list.append(tf.image.adjust_hue(
                    im[:, :, 3 * i: 3 * (i + 1)], hue_delta))
                #  tf.image.random_hue(im[:, :, 3 * i: 3 * (i + 1)], 0.1, seed=hue_seed))
            im = tf.concat(hue_im_list, axis=2)
            return im

        def random_augmentation(im):
            def augmentation_one(sim):
                do_aug = tf.random_uniform([], 0, 1)
                return tf.cond(do_aug > 0.5, lambda: augment_image_properties(sim), lambda: sim)
            im = tf.map_fn(lambda sim: augmentation_one(sim), im)
            #im = tf.cond(do_aug > 0.5, lambda: tf.map_fn(lambda sim: augment_image_properties(sim), im), lambda: im)
            return im

        im_all = random_flip(im_all)
        image = im_all[0]
        depth = im_all[1]
        image_aug = random_augmentation(image)
        return [image_aug, depth]

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
                                        frame_ids[i] + '.jpg') for i in range(len(frames))]
        depth_file_list = [os.path.join(data_root, subfolders[i],
                                        frame_ids[i] + '.png') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['depth_file_list'] = depth_file_list
        return all_list

    def unpack_image(self, image_seq, img_height, img_width, img_channel):
        image_seq.set_shape([img_height,
                             img_width,
                             img_channel])
        return image_seq
