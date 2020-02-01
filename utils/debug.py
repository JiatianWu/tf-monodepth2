import tensorflow as tf
import pdb

# img_reader = tf.WholeFileReader()
# _, image_contents = img_reader.read(image_paths_queue)
# image_seq = tf.image.decode_jpeg(image_contents, channels=3)

img_path = '/home/jiatian/dataset/tum/sequence_01/000001.jpg'
img = tf.read_file(img_path)
img_gray = tf.image.decode_jpeg(img)
img_v1 = tf.image.decode_jpeg(img, channels=3)
img_v2 = tf.image.grayscale_to_rgb(img_gray)

with tf.Session() as sess:
    np_img_v1 = sess.run(img_v1)
    np_img_v2 = sess.run(img_v2)

pdb.set_trace()