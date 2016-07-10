import tensorflow as tf
import numpy as np
import ray

@ray.remote(16 * [np.ndarray], [])
def updateweights(*weight):
  feed_dict = dict(zip(placeholders, weight))
  sess.run(assignment, feed_dict=feed_dict)

@ray.remote([np.ndarray, np.ndarray], 16 * [np.ndarray])
def computegrad(X, Y):
  return sess.run([g for (g,v) in compgrads], feed_dict={images:X, y_true:Y, dropout:0.5})

def printaccuracy(X, Y):
    print sess.run(accuracy, feed_dict={images:X, y_true:Y, dropout:1.0})

images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 1000])
parameters = []
assignment = []
placeholders = []
# conv1
with tf.name_scope('conv1') as scope:
  kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-1))
  conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
  biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                       trainable=True, name='biases')
  bias = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(bias, name=scope)
  parameters += [kernel, biases]

  kernel_new = tf.placeholder(tf.float32, shape=kernel.get_shape())
  biases_new = tf.placeholder(tf.float32, shape=biases.get_shape())
  update_kernel = kernel.assign(kernel_new)
  update_biases = biases.assign(biases_new)
  assignment += [update_kernel, update_biases]
  placeholders += [kernel_new, biases_new]
# pool1
pool1 = tf.nn.max_pool(conv1,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool1')


# lrn1
pool1lrn = tf.nn.lrn(pool1, depth_radius=5, bias=1.0,
                            alpha=0.0001, beta=0.75,
                            name="LocalResponseNormalization")
# conv2
with tf.name_scope('conv2') as scope:
  kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,
                                           stddev=1e-1), name='weights')
  conv = tf.nn.conv2d(pool1lrn, kernel, [1, 1, 1, 1], padding='SAME')
  biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                       trainable=True, name='biases')
  bias = tf.nn.bias_add(conv, biases)
  conv2 = tf.nn.relu(bias, name=scope)
  parameters += [kernel, biases]

  kernel_new = tf.placeholder(tf.float32, shape=kernel.get_shape())
  biases_new = tf.placeholder(tf.float32, shape=biases.get_shape())
  update_kernel = kernel.assign(kernel_new)
  update_biases = biases.assign(biases_new)
  assignment += [update_kernel, update_biases]
  placeholders += [kernel_new, biases_new]
pool2 = tf.nn.max_pool(conv2,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool2')

# lrn2
pool2lrn = tf.nn.lrn(pool2, depth_radius=5, bias=1.0,
                            alpha=0.0001, beta=0.75,
                            name="LocalResponseNormalization")
# conv3
with tf.name_scope('conv3') as scope:
  kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
  conv = tf.nn.conv2d(pool2lrn, kernel, [1, 1, 1, 1], padding='SAME')
  biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                       trainable=True, name='biases')
  bias = tf.nn.bias_add(conv, biases)
  conv3 = tf.nn.relu(bias, name=scope)
  parameters += [kernel, biases]

  kernel_new = tf.placeholder(tf.float32, shape=kernel.get_shape())
  biases_new = tf.placeholder(tf.float32, shape=biases.get_shape())
  update_kernel = kernel.assign(kernel_new)
  update_biases = biases.assign(biases_new)
  assignment += [update_kernel, update_biases]
  placeholders += [kernel_new, biases_new]
# conv4
with tf.name_scope('conv4') as scope:
  kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
  conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
  biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                       trainable=True, name='biases')
  bias = tf.nn.bias_add(conv, biases)
  conv4 = tf.nn.relu(bias, name=scope)
  parameters += [kernel, biases]
  kernel_new = tf.placeholder(tf.float32, shape=kernel.get_shape())
  biases_new = tf.placeholder(tf.float32, shape=biases.get_shape())
  update_kernel = kernel.assign(kernel_new)
  update_biases = biases.assign(biases_new)
  assignment += [update_kernel, update_biases]
  placeholders += [kernel_new, biases_new]
# conv5
with tf.name_scope('conv5') as scope:
  kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                           dtype=tf.float32,
                                           stddev=1e-1), name='weights')
  conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
  biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                       trainable=True, name='biases')
  bias = tf.nn.bias_add(conv, biases)
  conv5 = tf.nn.relu(bias, name=scope)
  parameters += [kernel, biases]
  kernel_new = tf.placeholder(tf.float32, shape=kernel.get_shape())
  biases_new = tf.placeholder(tf.float32, shape=biases.get_shape())
  update_kernel = kernel.assign(kernel_new)
  update_biases = biases.assign(biases_new)
  assignment += [update_kernel, update_biases]
  placeholders += [kernel_new, biases_new]

# pool5
pool5 = tf.nn.max_pool(conv5,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1],
                       padding='VALID',
                       name='pool5') 

# lrn5
pool5lrn = tf.nn.lrn(pool5, depth_radius=5, bias=1.0,
                            alpha=0.0001, beta=0.75,
                            name="LocalResponseNormalization")
dropout = tf.placeholder(tf.float32)

with tf.name_scope('fc1') as scope:
  n_input = int(np.prod(pool5lrn.get_shape().as_list()[1:]))
  kernel = tf.Variable(tf.truncated_normal([n_input, 4096], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
  fc_in = tf.reshape(pool5lrn, [-1, n_input])
  biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='biases')
  fc_layer1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(fc_in, kernel), biases))
  fc_out1 = tf.nn.dropout(fc_layer1, dropout)
  parameters += [kernel, biases]
  kernel_new = tf.placeholder(tf.float32, shape=kernel.get_shape())
  biases_new = tf.placeholder(tf.float32, shape=biases.get_shape())
  update_kernel = kernel.assign(kernel_new)
  update_biases = biases.assign(biases_new)
  assignment += [update_kernel, update_biases]
  placeholders += [kernel_new, biases_new]

with tf.name_scope('fc2') as scope:
  n_input = int(np.prod(fc_out1.get_shape().as_list()[1:]))
  kernel = tf.Variable(tf.truncated_normal([n_input, 4096], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
  fc_in = tf.reshape(fc_out1, [-1, n_input])
  biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                         trainable=True, name='biases')
  fc_layer2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(fc_in, kernel), biases))
  fc_out2 = tf.nn.dropout(fc_layer2, dropout)
  parameters += [kernel, biases]
  kernel_new = tf.placeholder(tf.float32, shape=kernel.get_shape())
  biases_new = tf.placeholder(tf.float32, shape=biases.get_shape())
  update_kernel = kernel.assign(kernel_new)
  update_biases = biases.assign(biases_new)
  assignment += [update_kernel, update_biases]
  placeholders += [kernel_new, biases_new]

with tf.name_scope('fc3') as scope:
  n_input = int(np.prod(fc_out2.get_shape().as_list()[1:]))
  kernel = tf.Variable(tf.truncated_normal([n_input, 1000], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
  fc_in = tf.reshape(fc_out2, [-1, n_input])
  biases = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32),
                         trainable=True, name='biases')
  fc_layer3 = tf.nn.softmax(tf.nn.bias_add(tf.matmul(fc_in, kernel), biases))
  parameters += [kernel, biases]
  kernel_new = tf.placeholder(tf.float32, shape=kernel.get_shape())
  biases_new = tf.placeholder(tf.float32, shape=biases.get_shape())
  update_kernel = kernel.assign(kernel_new)
  update_biases = biases.assign(biases_new)
  assignment += [update_kernel, update_biases]
  placeholders += [kernel_new, biases_new]

y_pred = fc_layer3 / tf.reduce_sum(fc_layer3,
                        reduction_indices=len(fc_layer3.get_shape()) - 1,
                        keep_dims=True)
# manual computation of crossentropy
y_pred = tf.clip_by_value(y_pred, tf.cast(1e-10, dtype=tf.float32),
                          tf.cast(1. - 1e-10, dtype=tf.float32))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred),
                                reduction_indices=len(y_pred.get_shape()) - 1))
#opt =  tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9) # aNY OTHER OPTIMIZIER CAN BE PLACED HERE
correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

compgrads = opt.compute_gradients(cross_entropy, parameters)

application = opt.apply_gradients(zip(placeholders,parameters))
sess = tf.Session()
sess.run(tf.initialize_all_variables())
