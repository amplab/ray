import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

def recurrent():     
    lr = 1.0
    titers = 1500
    batch1 = 128
    displaystep = 10
    n_in = 28
    n_steps = 28
    n_hidden = 28
    n_classes = 10
    x2 = tf.placeholder("float", [None, n_steps, n_in])
    y2 = tf.placeholder("float", [None, n_classes])
    init_state = tf.placeholder("float", [None, 2*n_in])
    W = tf.Variable(tf.random_normal([n_hidden, n_classes]))
    B = tf.Variable(tf.random_normal([10]))    
    XT=tf.transpose(x2, [1,0,2])
    XR=tf.reshape(XT, [-1, n_in])
    X_split = tf.split(0, n_steps, XR)
    lstm = rnn_cell.BasicLSTMCell(n_hidden)
    outputs, state = rnn.rnn(lstm, X_split, initial_state = init_state)
    pred = tf.matmul(outputs[-1], W) + B
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y2))
    optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    corpred= tf.equal(tf.argmax(pred,1), tf.argmax(y2,1))
    with tf.Session() as sess:	 
         sess.run(tf.initialize_all_variables())
         for i in range(titers):
	     batchxs, batchys = mnist.train.next_batch(batch1)
             batchxs = batchxs.reshape((batch1, n_steps, n_in))
	     sess.run(optimizer, feed_dict={x2: batchxs, y2: batchys, init_state: np.zeros((batch1, lstm.state_size))})
	     if i%100 ==0:
                print(i, np.mean(sess.run(corpred,feed_dict={x2: batchxs, y2:batchys, init_state: np.zeros((batch1, lstm.state_size))})))	
         print(sess.run(tf.reduce_mean(tf.cast(corpred, tf.float32)),feed_dict={x2:mnist.test.images.reshape(-1,28,28), y2:mnist.test.labels, init_state: np.zeros((10000,lstm.state_size))}))
    return

def convolutional():
    x = tf.placeholder(tf.float32, shape=[None,784])
    y = tf.placeholder(tf.float32, shape = [None, 10])
    W_conv1 = weight([5,5,1,32])
    B_conv1 = bias([32])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + B_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight([5,5,32,64])
    b_conv2 = bias([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight([7*7*64, 1024])
    b_fc1 = bias([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop= tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight([1024, 10])
    b_fc2 = bias([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    with tf.Session() as sess:
         sess.run(tf.initialize_all_variables())
         for i in range(10000):
	     batch = mnist.train.next_batch(50)
            
	     if i%100 == 0:
	        train_ac = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob:0.5})
                print("step %d, training accuracy	 %g"%(i, train_ac))
         print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))
    return

recurrent()
