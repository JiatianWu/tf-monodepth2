#Copyright 2020 Nod Labs
import argparse
import open3d as o3d
import os
import json
import glob
import sys
import pickle
from PIL import Image
import numpy as np

class KinectDataSource:

    def __init__(self, input, output):
        self.input = input
        self.output = output

        self.crop_corner = (160, 0) # (x, y)
        self.crop_size = (960, 720) # (width, height)
        self.output_size = (640, 480) # (width, height)

        self.zoom_x = float(self.output_size[0] / self.crop_size[0])
        self.zoom_y = float(self.output_size[1] / self.crop_size[1])

        self.reader = o3d.io.AzureKinectMKVReader()
        self.reader.open(self.input)
        if not self.reader.is_opened():
            raise RuntimeError("Unable to open file {}".format(args.input))

    def concat_image_seq(self, seq):
        for i, image in enumerate(seq):
            if i == 0:
                res = image
            else:
                res = np.hstack((res, image))
        return res

    def load_intrinsics_raw(self, metadata_dict):
        intrinsic_matrix = metadata_dict['intrinsic_matrix']
        P = np.transpose(np.reshape(intrinsic_matrix, (3, 3)))
        return P

    def crop_intrinsics(self, mat):
        out = np.copy(mat)
        out[0,2] -= self.crop_corner[0]
        out[1,2] -= self.crop_corner[1]
        return out

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out

    def write_intrinsics(self, intrinsics, path):
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        with open(path, 'w') as f:
            f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

    def generate_train_txt(self):
        with open(self.output + '/train.txt', 'w') as tf:
            train_data_dir = self.output + '/train_data'
            frame_ids = os.listdir(train_data_dir)
            for frame in frame_ids:
                if '.jpg' in frame:
                    tf.write('%s %s\n' % ('train_data', frame[:-4]))

    def preprocess_image(self, image):
        color_image = Image.fromarray(image)
        color_image = color_image.crop(box=(self.crop_corner[0],
                                            self.crop_corner[1],
                                            self.crop_corner[0] + self.crop_size[0],
                                            self.crop_corner[1] + self.crop_size[1]))
        color_image = color_image.resize(self.output_size)

        return color_image

    def run_rgb_only(self):
        idx = 0
        while not self.reader.is_eof():
            rgbd = self.reader.next_frame()
            if rgbd is None:
                continue

            if self.output is not None:
                color_image = self.preprocess_image(np.asarray(rgbd.color))
                color_filename = '{0}/train_data/{1:06d}.jpg'.format(
                    self.output, idx)
                print('Writing to {}'.format(color_filename))
                color_image.save(color_filename)
                idx += 1

        self.reader.close()

    def run(self):
        if self.output is not None:
            abspath = os.path.abspath(self.output)
            metadata = self.reader.get_metadata()
            o3d.io.write_azure_kinect_mkv_metadata(
                '{}/intrinsic.json'.format(abspath), metadata)
            with open('{}/intrinsic.json'.format(abspath)) as json_file:
                metadata_dict = json.load(json_file)

        intrinsics_raw = self.load_intrinsics_raw(metadata_dict)
        intrinsics_crop = self.crop_intrinsics(intrinsics_raw)
        intrinsics_scale = self.scale_intrinsics(intrinsics_crop, self.zoom_x, self.zoom_y)

        idx = 0
        image_seq = []
        depth_seq = []
        while not self.reader.is_eof():
            rgbd = self.reader.next_frame()
            if rgbd is None:
                continue

            if self.output is not None:
                color_image = self.preprocess_image(np.asarray(rgbd.color))
                depth_image = self.preprocess_image(np.asarray(rgbd.depth))
                if len(image_seq) < 2:
                    image_seq.append(np.array(color_image))
                    depth_seq.append(np.array(depth_image))
                else:
                    color_filename = '{0}/train_data/{1:06d}.jpg'.format(
                        self.output, idx)
                    print('Writing to {}'.format(color_filename))
                    image_seq.append(np.array(color_image))
                    tosave_image_seq = self.concat_image_seq(image_seq)
                    Image.fromarray(tosave_image_seq).save(color_filename)

                    depth_filename = '{0}/depth_data/{1:06d}.pkl'.format(
                                     self.output, idx)
                    print('Writing to {}'.format(depth_filename))
                    depth_seq.append(np.array(depth_image))
                    tosave_depth_seq = self.concat_image_seq(depth_seq)
                    data_dict = {'depth_gt': tosave_depth_seq}
                    output = open(depth_filename, 'wb')
                    pickle.dump(data_dict, output)
                    output.close()

                    intrinsics_filename = '{0}/train_data/{1:06d}_cam.txt'.format(self.output, idx)
                    self.write_intrinsics(intrinsics_scale, intrinsics_filename)
                    idx += 1
                    image_seq.pop(0)
                    depth_seq.pop(0)

        self.reader.close()

        self.generate_train_txt()

if __name__ == '__main__':
    import time
    parser = argparse.ArgumentParser(description='Azure kinect mkv reader.')
    parser.add_argument('--input',
                        type=str,
                        required=True,
                        help='input mkv file')
    parser.add_argument('--output',
                        type=str,
                        help='output path to store color/ and depth/ images')
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        exit()

    args.output += time.ctime()
    if args.output is None:
        print('No output path, only play mkv')
    else:
        try:
            os.mkdir(args.output)
            os.mkdir('{}/train_data'.format(args.output))
            os.mkdir('{}/depth_data'.format(args.output))
        except (PermissionError, FileExistsError):
            print('Unable to mkdir {}, only play mkv'.format(args.output))
            args.output = None

    reader = KinectDataSource(args.input, args.output)
    reader.run()
