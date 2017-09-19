from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()

#input layer units
in_units = 784

#hidden layer units
h1_units = 300

# hidden_layer
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))

# output layer
W2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

# input variable
x = tf.placeholder(tf.float32, [None, in_units])

# Dropout --  keep probability
keep_prob = tf.placeholder(tf.float32)

# hidden layer
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

# output layer
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# expect output layer
y_ = tf.placeholder(tf.float32, [None, 10])

# cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis = 1))

# use the Adagrad optimizer
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# Begian to train
tf.global_variables_initializer().run()

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob:0.75})

# correct prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))