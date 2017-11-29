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
        # Inception Block One -- includes three inception modules (Mixed_5b/5c/5d)
        # Mixed_5b
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1,1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_0, 64, [5,5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1,1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3,3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3,3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1,1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0,branch_1,branch_2,branch_3], 3)

