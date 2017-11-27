import tensorflow as tf

slim = tf.contrib.slim

# lambda function to create a normal distribution
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# function to generate the default parameters' values
def inception_v3_arg_scope(weight_decay=0.00004,
                           stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
    batch_norm_var_collection = {
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
                normalizer_params=batch_norm_var_collection) as sc:
            return sc
