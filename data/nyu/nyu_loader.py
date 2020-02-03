from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc

class nyu_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split,
                 sequence_id,
                 img_height=256,
                 img_width=256,
                 seq_length=5):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.date_list = [os.listdir(self.dataset_dir)[sequence_id]]
        self.collect_train_frames()
        
    def collect_train_frames(self):
        all_frames = []
        for date in self.date_list:
            print('Processing' + date)
            img_dir = self.dataset_dir + date
            N = len(glob(img_dir + '/*.ppm'))

            frame_list = sorted(os.listdir(img_dir))
            for frame in frame_list:
                if '.ppm' in frame:
                    frame_id = frame
                    all_frames.append(date + ' ' + frame_id)

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, fid = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, min_src_fid = frames[min_src_idx].split(' ')
        max_src_drive, max_src_fid = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
            return True
        return False

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset + 1):
            curr_idx = tgt_idx + o
            curr_drive, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image_raw(curr_drive, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics_raw(tgt_drive, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        return example

    def load_image_raw(self, drive, frame_id):
        img_file = self.dataset_dir + drive + '/' + frame_id + '.ppm'
        img = scipy.misc.imread(img_file)
        return img

    def load_intrinsics_raw(self, drive, frame_id):
        P_rect = np.eye(3, 3)
        P_rect[0,0] = 5.1885790117450188e+02
        P_rect[0,2] = 3.2558244941119034e+02
        P_rect[1,1] = 5.1946961112127485e+02
        P_rect[1,2] = 2.5373616633400465e+02
        return P_rect

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out