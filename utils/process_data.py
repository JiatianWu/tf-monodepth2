import os
import glog
import pdb
import pickle
import tensorflow as tf

def debug_convert():
    img_path = '/home/jiatian/dataset/tum/sequence_01/000001.jpg'
    img = tf.read_file(img_path)
    img_gray = tf.image.decode_jpeg(img)
    img_v1 = tf.image.decode_jpeg(img, channels=3)
    img_v2 = tf.image.grayscale_to_rgb(img_gray)

    with tf.Session() as sess:
        np_img_v1 = sess.run(img_v1)
        np_img_v2 = sess.run(img_v2)

def read_imu_data():
    data_path = '/home/jiatian/dataset/office_kitchen_0001a/a-1315403270.593277-3849079182.dump'
    # with open(data_path, 'rb') as f:
    #     data = pickle.load(f)
    f = open(data_path, 'r')
    pdb.set_trace()

def resave_imu_data():
    dataset = '/freezer/nyudepthV2_raw'
    seqs = os.listdir(dataset)
    for seq in seqs:
        seq_dir = dataset + '/' + seq
        for data in os.listdir(seq_dir):
            if data[0] == 'a':
                imu_data_path = seq_dir + '/' + data
                resave_imu_data_path = seq_dir + '/' + data[:-4] + '.txt'
                call_resave_imu(imu_data_path, resave_imu_data_path)

def call_resave_imu(orig_path, resave_path):
    command = './resave_imu ' + orig_path + ' ' + resave_path
    os.system(command)

if __name__ == "__main__":
    resave_imu_data()