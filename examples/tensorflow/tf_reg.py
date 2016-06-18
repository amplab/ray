import tensorflow as tf
import numpy as np

def lin_reg(X,Y):
    model = tf.add(tf.matmul(X, w),b)
    cost = tf.reduce_sum(tf.square(Y-model))/(2*n) 
    return tf.train.GradientDescentOptimizer(0.02).minimize(cost)

def log_reg(X,Y):
    y = tf.nn.softmax(tf.matmul(X,w) + b)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y* tf.log(y), reduction_indices = [1]))
    return tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
def linregtest(w,b):
    print np.linalg.norm(2 * (inX.T.dot(inX.dot(np.array(w).reshape((d,1))) + np.array(b).reshape((n,1))) - inX.T.dot(inY)))
    return

n = 100000
d = 400
inX1 = np.random.random((n,d))
inX = np.random.random((n,d))
inY = np.random.random((n,1))
#inYlog = np.where(inX.dot(np.array([0,4,1,2])) >= 1.5, 1, 0).reshape((n,1))
#inYlog1 = np.where(inX1.dot(np.array([0,4,1,2])) >= 1.5, 1, 0).reshape((n,1))
X = tf.placeholder('float', [None, d], name = "X")
Y = tf.placeholder('float', [None,1], name="Y")
w = tf.Variable(tf.random_normal([d,1]))
b = tf.Variable(tf.random_normal([n,1]))
cur_reg = lin_reg(X,Y)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
     sess.run(init_op)
     for i in range(100):
	     sess.run(cur_reg, feed_dict={X:inX,Y:inY})
     linregtest(sess.run(w), sess.run(b))
    #print np.sum(inYlog1.reshape(n,) == np.squeeze(sess.run(tf.to_int32(tf.nn.softmax(tf.matmul(X,w)+b)), feed_dict={X:inX1})))
    #print np.sum(inYlog.reshape(n,) == np.squeeze(sess.run(tf.to_int32(tf.nn.softmax(tf.matmul(X,w)+b)), feed_dict={X:inX})))
