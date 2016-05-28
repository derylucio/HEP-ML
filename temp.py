print labels[1:20, :]
tf.reset_default_graph()
sess = tf.InteractiveSession()
flattened = np.prod(data_samples.shape[1:])
aug_data = data_samples.reshape((data_samples.shape[0], data_samples.shape[1], data_samples.shape[2], 1))
batch_size = 35
dropout_prob = 0.9
num_train = 980 #int(np.round(aug_data.shape[0] * 0.7))
num_batches = num_train / batch_size 


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
keep_prob = tf.placeholder(tf.float32) #adding drop out 


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
W_fc1 = weight_variable([final_size* 64,  1024])#1024
b_fc1 = bias_variable([1024]) #1024

h_pool2_flat = tf.reshape(h_pool2, [-1, final_size*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#READ OUT LAYER
W_fc2 = weight_variable([1024, 2]) #1024
b_fc2 = bias_variable([2])
intermediate = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(intermediate)
# cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv)))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(intermediate, y))
#max_margin_loss = tf.reduce_mean(tf.maximum( 0.0, (1.0 - y)*y_conv - y*y_conv + 1.0 ))
loss = tf.reduce_mean(tf.square(tf.maximum( 0.0, (1.0 - y)*y_conv - y*y_conv + 1.0 )))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#accuracy metric
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
for epoch in range(2):
	print("This is a new epoch %d"%epoch)
	for i in range(num_batches):
		examples = aug_data[(i*batch_size):((i + 1)*batch_size), :, :, :]
		divs =  labels[(i*batch_size):((i + 1)*batch_size), :]
		if i % 6 == 0:
			train_accuracy = acc.eval(feed_dict={
		        x_img:examples, y: divs, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
			train_loss = sess.run([loss], feed_dict={
		        x_img:examples, y: divs, keep_prob: 1.0})
			print("step %d, loss %g"%(i, train_loss[0]))
		train_step.run(feed_dict={x_img : examples, y : divs, keep_prob:dropout_prob})

examples = aug_data[(num_train + 1):, :, :, :]
classes = labels[(num_train + 1):, :]

# print("test accuracy %g"%acc.eval(feed_dict={
#     x_img: examples, y: classes, keep_prob: 1.0}))
predictions_scores = y_conv.eval(feed_dict={x_img : examples, keep_prob:1.0})
predictions_scores1 = np.array(sess.run([y_conv], feed_dict={x_img : examples, keep_prob:1.0}))
flattened_classes = np.array(classes.argmax(axis=1))
indices = np.arange(flattened_classes.shape[0])
# predictions_scores = np.squeeze(np.array(predictions_scores), axis=(0,))
print num_train
print np.sum(predictions_scores.argmax(axis=1) != 1)
print np.sum(flattened_classes)
print 'Old Pred scores'
print predictions_scores[0:10,:]
print 'New Pred Scores'
print predictions_scores1[0:10,:]
corresponding_scores = predictions_scores[indices, flattened_classes]
print corresponding_scores[0:10]
fpr, tpr , thresholds = roc_curve(flattened_classes, corresponding_scores)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
print fpr[0:10]
print tpr[0:10]
area = auc(fpr, tpr)#np.trapz(efficiencies[:,1], x=efficiencies[:,0])
plt.plot(fpr, tpr)
title = "CNN Graph"
plt.figtext(.4, .5, "AUC : " + str(area))
auc = roc_auc_score(classes, predictions_scores)
print 'Here is the roc_auc_score 1'
print auc
print 'Here is the Area 2'
print area
plt.show()
sess.close()