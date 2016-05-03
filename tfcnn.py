import tensorflow as tf
import numpy as np 
from extract_data_from_trees import extract_imagedata

data_samples, labels = extract_imagedata()
tf.reset_default_graph()
sess = tf.InteractiveSession()
#First build dummy softmax regression model
flattened = np.prod(data_samples.shape[1:])
print data_samples.shape
aug_data = data_samples.reshape((data_samples.shape[0], data_samples.shape[1], data_samples.shape[2], 1))
print aug_data.shape
#x = tf.placeholder(tf.float32, shape=[None, flattened])
#NEED TO FLATTEN DATA MATRIX IF USING THIS
# W = tf.Variable(tf.zeros([flattened, 2]))
# b = tf.Variable(tf.zeros([2]))
# sess.run(tf.initialize_all_variables())
# predictions = tf.nn.softmax(tf.matmul(x, W) + b)
# print predictions
# cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y* tf.log(predictions), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy_loss)
batch_size = 40
num_train = int(np.round(aug_data.shape[0] * 0.75))
num_batches = num_train / batch_size 
# for i in range(num_batches):
# 	examples = aug_data[(i*batch_size):((i + 1)*batch_size), :]
# 	divs =  labels[(i*batch_size):((i + 1)*batch_size), :]
# 	train_step.run(feed_dict={x : examples, y : divs})
# correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(predictions, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) #convert booleans in correct predictions to float32
# examples = aug_data[num_train:, :]
# classes = labels[num_train:, :]
# print 'Softmax Accuracy : ', accuracy.eval(feed_dict={x:examples, y:classes})
# # Accuracy of 22% :(


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#FIRST CONVOLUTIONAL LAYER
x_img   = tf.placeholder(tf.float32, shape=(None,data_samples.shape[1], data_samples.shape[2], 1))
y = tf.placeholder(tf.float32, shape=(None, 2))

W_conv1 = weight_variable([11, 11, 1, 32])
b_conv1 = bias_variable([32])
hconv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
h_pool1 = maxpool_2x2(hconv1)
#SECOND CONVOLUTIONAL LAYER
W_conv2 = weight_variable([11, 11, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = maxpool_2x2(h_conv2)
#FULLY CONNECTED LAYER
final_size = 208 #float(np.floor((data_samples.shape[1]*data_samples.shape[2])/16))
W_fc1 = weight_variable([final_size* 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, final_size*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32) #adding drop out 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#READ OUT LAYER
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv)))
train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cross_entropy_loss)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())


for i in range(num_batches):
	examples = aug_data[(i*batch_size):((i + 1)*batch_size), :, :, :]
	divs =  labels[(i*batch_size):((i + 1)*batch_size), :]
	if i % 5 == 0:
		train_accuracy = acc.eval(feed_dict={
	        x_img:examples, y: divs, keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x_img : examples, y : divs, keep_prob:0.5})

examples = aug_data[num_train:, :, :, :]
classes = labels[num_train:, :]
print("test accuracy %g"%acc.eval(feed_dict={
    x_img: examples, y: classes, keep_prob: 1.0}))


# #Original test on mnist dataset.
# import tensorflow as tf
# import numpy as np 
# from extract_data_from_trees import extract_imagedata

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# tf.reset_default_graph()

# sess = tf.InteractiveSession()

# def weight_variable(shape):
# 	initial = tf.truncated_normal(shape, stddev=0.1)
# 	return tf.Variable(initial)

# def bias_variable(shape):
# 	initial = tf.constant(0.1, shape=shape)
# 	return tf.Variable(initial)

# def conv2d(x, W):
# 	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# def maxpool_2x2(x):
# 	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# #FIRST CONVOLUTIONAL LAYER
# x = tf.placeholder(tf.float32, shape=[None, 784])
# x_img   = tf.reshape(x, [-1, 28,28,1])
# y = tf.placeholder(tf.float32, shape=[None, 10])


# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])
# hconv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
# h_pool1 = maxpool_2x2(hconv1)
# #SECOND CONVOLUTIONAL LAYER
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = maxpool_2x2(h_conv2)
# #FULLY CONNECTED LAYER
# final_size = 49 #float(np.floor((data_samples.shape[1]*data_samples.shape[2])/16))
# W_fc1 = weight_variable([final_size* 64, 1024])
# b_fc1 = bias_variable([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, final_size*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# keep_prob = tf.placeholder(tf.float32) #adding drop out 
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# #READ OUT LAYER
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# print y * tf.log(y_conv)
# cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv)))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# sess.run(tf.initialize_all_variables())


# for i in range(200):
#   batch = mnist.train.next_batch(50)
#   if i%100 == 0:
#     train_accuracy = acc.eval(feed_dict={
#         x:batch[0], y: batch[1], keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#   train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

# print("test accuracy %g"%acc.eval(feed_dict={
#     x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))




