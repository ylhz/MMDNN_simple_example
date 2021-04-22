# coding=utf-8
"""Save model with .meta and .data"""

import os
import tensorflow as tf
from nets import inception_v3
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"


slim = tf.contrib.slim
tf.flags.DEFINE_string('checkpoint_path', './models/inception_v3.ckpt',
                       'Path to checkpoint for pretained models.')
tf.flags.DEFINE_string('output_path', 'output_model/inception_v3.ckpt',
                       'Path to checkpoint for output models.')                       
FLAGS = tf.flags.FLAGS


model_weight_path = FLAGS.checkpoint_path
# model_weight_path = './model/adv_inception_v3_rename.ckpt'

if __name__ == '__main__':

    batch_shape = [None, 299, 299, 3]
    num_classes = 1001

    # with tf.Graph().as_default():
    tf.reset_default_graph()
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits_v3, end_points_v3 = inception_v3.inception_v3(
                x_input, num_classes=num_classes, is_training=False)
    flow = tf.identity(logits_v3, name="MMdnn_Output") # set output node name
    s = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        s.restore(sess, model_weight_path)
        # model_path = os.path.join(FLAGS.output_path + 'inception_v3.ckpt')
        s.save(sess,  FLAGS.output_path)
        # s.save(sess, model_path)
        print('done!')
        print('model save in {}', FLAGS.output_path)


