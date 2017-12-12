import tensorflow as tf

slim = tf.contrib.slim

# lambda function to create a normal distribution
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# function to generate the default parameters' values
def inception_v3_arg_scope(weight_decay=0.00004,
                           stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }

    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weight_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params) as sc:
            return sc


# function to generate the CNN part
def inception_v3_base(inputs, scope=None):

    '''
      Here is a mapping from the old_names to the new names:
      Old name          | New name
      =======================================
      conv0             | Conv2d_1a_3x3
      conv1             | Conv2d_2a_3x3
      conv2             | Conv2d_2b_3x3
      pool1             | MaxPool_3a_3x3
      conv3             | Conv2d_3b_1x1
      conv4             | Conv2d_4a_3x3
      pool2             | MaxPool_5a_3x3
      mixed_35x35x256a  | Mixed_5b
      mixed_35x35x288a  | Mixed_5c
      mixed_35x35x288b  | Mixed_5d
      mixed_17x17x768a  | Mixed_6a
      mixed_17x17x768b  | Mixed_6b
      mixed_17x17x768c  | Mixed_6c
      mixed_17x17x768d  | Mixed_6d
      mixed_17x17x768e  | Mixed_6e
      mixed_8x8x1280a   | Mixed_7a
      mixed_8x8x2048a   | Mixed_7b
      mixed_8x8x2048b   | Mixed_7c
    '''

    end_points = {}

    # CNN layers and Pool layers ahead of Inception module
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
            # 299x299x3
            net = slim.conv2d(inputs, 32, [3,3], stride=2, scope='Conv2d_1a_3x3')
            # 149x149x32
            net = slim.conv2d(net, 32, [3,3], stride=1, scope='Conv2d_2a_3x3')
            # 147x147x32
            net = slim.conv2d(net, 64, [3,3], padding='SAME', scope='Conv2d_2b_3x3')
            # 147x147x64
            net = slim.max_pool2d(net, [3,3], stride=2, scope='MaxPool_3a_3x3')
            # 73x73x64
            net = slim.conv2d(net, 80, [1,1], scope='Conv2d_3b_1x1')
            # 73x73x80
            net = slim.conv2d(net, 192, [3,3], scope='Conv2d_4a_3x3')
            # 71x71x192
            net = slim.max_pool2d(net, [3,3], stride=2, scope='MaxPool_5a_3x3')
            # 35*35*192

        # Inception Blocks
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            # Inception Block One -- includes three inception modules (Mixed_5b/5c/5d)
            # Mixed_5b  -- 35x35x256
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')       # 35x35x192 --> 35x35x64
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1x1')       # 35x35x192 --> 35x35x48
                    branch_1 = slim.conv2d(branch_0, 64, [5,5], scope='Conv2d_0b_5x5')  # 35x35x48  --> 35x35x64
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')       # 35x35x192 --> 35x35x64
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3x3')  # 35x35x64  --> 35x35x96
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3x3')  # 35x35x96  --> 35x35x96
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')      # 35x35x192 --> 35x35x192
                    branch_3 = slim.conv2d(branch_3, 32, [1,1], scope='Conv2d_0b_1x1')  # 35x35x192 --> 35x35x32

                net = tf.concat([branch_0,branch_1,branch_2,branch_3], 3)  # 35x35x(64+64+96+32) = 35x35x256

            # Mixed_5c -- 35x35x288
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')       # 35x35x256 --> 35x35x64
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1x1')       # 35x35x256 --> 35x35x48
                    branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_5x5')  # 35x35x48  --> 35x35x64
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')       # 35x35x256 --> 35x35x64
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3x3')  # 35x35x64  --> 35x35x96
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3x3')  # 35x35x96  --> 35x35x96
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')      # 35x35x256 --> 35x35x256
                    branch_3 = slim.conv2d(branch_3, 64, [1,1], scope='Conv2d_0b_1x1')  # 35x35x256 --> 35x35x64

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)  # 35x35x(64+64+96+64) = 35x35x288

            # Mixed_5d -- 35x35x288 paths
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')       # 35x35x288 --> 35x35x64
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1x1')       # 35x35x288 --> 35x35x48
                    branch_1 = slim.conv2d(branch_1, 64, [5,5], scope='Conv2d_0b_5x5')  # 35x35x48  --> 35x35x64
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')       # 35x35x288 --> 35x35x64
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3x3')  # 35x35x64  --> 35x35x96
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3x3')  # 35x35x96  --> 35x35x96
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')      # 35x35x288 --> 35x35x288
                    branch_3 = slim.conv2d(branch_3, 64, [1,1], scope='Conv2d_0b_1x1')  # 35x35x288 --> 35x35x64

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)  # 35x35x(64+64+96+64) = 35x35x288

            # Inception Block Two -- 5 inception modules (Mixed_6a/6b/6c/6d/6e)
            # Mixed_6a -- 17x17x768
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3,3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')   # 35x35x288 --> 17x17x384
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')                               # 35x35x288 --> 35x35x64
                    branch_1 = slim.conv2d(branch_1, 96, [3,3], scope='Conv2d_0b_3x3')                          # 35x35x64  --> 35x35x96
                    branch_1 = slim.conv2d(branch_1, 96, [3,3], stride=2, padding='VALID', scope='Con2d_1a_3x3')# 35x35x96  --> 17x17x96
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3,3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')   # 35x35x288 --> 17x17x288

                net = tf.concat([branch_0, branch_1, branch_2], 3)      # 17x17x(384+96+288) = 17x17x768

            # Mixed_6b -- 17x17x768
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x192
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x128
                    branch_1 = slim.conv2d(branch_1, 128, [1,7], scope='Conv2d_0b_1x7')     # 17x17x128 --> 17x17x128
                    branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7x1')     # 17x17x128 --> 17x17x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x128
                    branch_2 = slim.conv2d(branch_2, 128, [7,1], scope='Conv2d_0b_7x1')     # 17x17x128 --> 17x17x128
                    branch_2 = slim.conv2d(branch_2, 128, [1,7], scope='Conv2d_0c_1x7')     # 17x17x128 --> 17x17x128
                    branch_2 = slim.conv2d(branch_2, 128, [7,1], scope='Conv2d_0d_7x1')     # 17x17x128 --> 17x17x128
                    branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1x7')     # 17x17x128 --> 17x17x192
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')          # 17x17x768 --> 17x17x768
                    branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1x1')     # 17x17x768 --> 17x17x192

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)    #17x17x(192+192+192+192) = 17x17x768

            # Mixed_6c -- 17x17x768
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x192
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x160
                    branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='Conv2d_0b_1x7')     # 17x17x160 --> 17x17x160
                    branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7x1')     # 17x17x160 --> 17x17x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x160
                    branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0b_7x1')     # 17x17x160 --> 17x17x160
                    branch_2 = slim.conv2d(branch_2, 160, [1,7], scope='Conv2d_0c_1x7')     # 17x17x160 --> 17x17x160
                    branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0d_7x1')     # 17x17x160 --> 17x17x160
                    branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1x7')     # 17x17x160 --> 17x17x192
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')          # 17x17x768 --> 17x17x768
                    branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1x1')     # 17x17x768 --> 17x17x192

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)    #17x17x(192+192+192+192) = 17x17x768

            # Mixed_6d -- 17x17x768
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x192
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x160
                    branch_1 = slim.conv2d(branch_1, 160, [1,7], scope='Conv2d_0b_1x7')     # 17x17x160 --> 17x17x160
                    branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7x1')     # 17x17x160 --> 17x17x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x160
                    branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0b_7x1')     # 17x17x160 --> 17x17x160
                    branch_2 = slim.conv2d(branch_2, 160, [1,7], scope='Conv2d_0c_1x7')     # 17x17x160 --> 17x17x160
                    branch_2 = slim.conv2d(branch_2, 160, [7,1], scope='Conv2d_0d_7x1')     # 17x17x160 --> 17x17x160
                    branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1x7')     # 17x17x160 --> 17x17x192
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')          # 17x17x768 --> 17x17x768
                    branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1x1')     # 17x17x768 --> 17x17x192

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)    #17x17x(192+192+192+192) = 17x17x768

            # Mixed_6e -- 17x17x768
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x192
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x192
                    branch_1 = slim.conv2d(branch_1, 192, [1,7], scope='Conv2d_0b_1x7')     # 17x17x192 --> 17x17x192
                    branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7x1')     # 17x17x192 --> 17x17x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x192
                    branch_2 = slim.conv2d(branch_2, 192, [7,1], scope='Conv2d_0b_7x1')     # 17x17x192 --> 17x17x192
                    branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0c_1x7')     # 17x17x192 --> 17x17x192
                    branch_2 = slim.conv2d(branch_2, 192, [7,1], scope='Conv2d_0d_7x1')     # 17x17x192 --> 17x17x192
                    branch_2 = slim.conv2d(branch_2, 192, [1,7], scope='Conv2d_0e_1x7')     # 17x17x192 --> 17x17x192
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')          # 17x17x768 --> 17x17x768
                    branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1x1')     # 17x17x768 --> 17x17x192

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)    #17x17x(192+192+192+192) = 17x17x768

            end_points['Mixed_6e'] = net

            # Inception Block Three -- 3 inception modules (Mixed_7a/7b/7c)
            # Mixed_7a -- 8x8x1280
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x192
                    branch_0 = slim.conv2d(branch_0, 320, [3,3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')          # 17x17x192 --> 8x8x320
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1,1], scope='Conv2d_0a_1x1')          # 17x17x768 --> 17x17x192
                    branch_1 = slim.conv2d(branch_1, 192, [1,7], scope='Conv2d_0b_1x7')     # 17x17x192 --> 17x17x192
                    branch_1 = slim.conv2d(branch_1, 192, [7,1], scope='Conv2d_0c_7x1')     # 17x17x192 --> 17x17x192
                    branch_1 = slim.conv2d(branch_1, 192, [3,3], stride=2,
                                           padding='VALID', scope='Con2d_1a_3x3')           # 17x17x192 --> 8x8x192
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3,3], stride=2,
                                               padding='VALID', scope='MaxPool_1a_3x3')     # 17x17x768 --> 8x8x768

                net = tf.concat([branch_0, branch_1, branch_2], 3)       # 8x8x(320+192+768) = 8x8x1280

            # Mixed_7b -- 8x8x2048
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1,1], scope='Conv2d_0a_1x1')          # 8x8x1280 --> 8x8x320
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1,1], scope='Conv2d_0a_1x1')          # 8x8x1280 --> 8x8x384
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1,3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3,1], scope='Conv2d_0b_3x1')], 3)       # 8x8x384  --> 8x8x768
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1,1], scope='Conv2d_0a_1x1')          # 8x8x1280 --> 8x8x448
                    branch_2 = slim.conv2d(branch_2, 384, [3,3], scope='Conv2d_0b_3x3')     # 8x8x448  --> 8x8x384
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1,3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3,1], scope='Conv2d_0c_3x1')], 3)       # 8x8x384  --> 8x8x768
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')          # 8x8x1280 --> 8x8x1280
                    branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1x1')     # 8x8x1280 --> 8x8x192

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)       # 8x8x(320+768+768+192) = 8x8x2048

            # Mixed_7c -- 8x8x2048
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1,1], scope='Conv2d_0a_1x1')          # 8x8x2048 --> 8x8x320
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1,1], scope='Conv2d_0a_1x1')          # 8x8x2048 --> 8x8x384
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1,3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3,1], scope='Conv2d_0b_3x1')], 3)       # 8x8x384  --> 8x8x768
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1,1], scope='Conv2d_0a_1x1')          # 8x8x2048 --> 8x8x448
                    branch_2 = slim.conv2d(branch_2, 384, [3,3], scope='Conv2d_0b_3x3')     # 8x8x448  --> 8x8x384
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1,3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3,1], scope='Conv2d_0c_3x1')], 3)       # 8x8x384  --> 8x8x768
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')          # 8x8x2048 --> 8x8x2048
                    branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='Conv2d_0b_1x1')     # 8x8x2048 --> 8x8x192

                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)       # 8x8x(320+768+768+192) = 8x8x2048

            return net, end_points


# Inception V3 function
def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):

    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:

        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):

            net, end_points = inception_v3_base(inputs, scope=scope)



