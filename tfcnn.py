import tensorflow as tf
import numpy as np 
from extract_data_from_trees import extract_imagedata
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt 
import sys
from matplotlib.backends.backend_pdf import PdfPages


#Iterate through the dataset
def data_iterator(orig_X, orig_y=None, batch_size=32, label_size=2, shuffle=False):
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(orig_X))
    data_X = orig_X[indices]
    data_y = orig_y[indices] if np.any(orig_y) else None
  else:
    data_X = orig_X
    data_y = orig_y
  ###
  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    x = data_X[batch_start:batch_start + batch_size]
    # Convert our target from the class index to a one hot vector
    y = None
    if np.any(data_y):
    	y =  data_y[batch_start:batch_start + batch_size] #None
    #   y_indices = data_y[batch_start:batch_start + batch_size]
    #   y = np.zeros((len(x), label_size), dtype=np.int32)
    #   y[np.arange(len(y_indices)), y_indices] = 1
    ###
    yield x, y
    total_processed_examples += len(x)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)


class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  DIM_ETA = 52
  DIM_PHI = 64
  num_channels = 1
  num_classes = 2
  dropout = 0.9
  lr = 1e-4
  final_size = 208
  batch_size = 128
  num_train = 1024#2432
  max_epoch = 20
  early_stopping = 2

class HEPModel(object):

	def load_data(self):
	    """Loads data from disk and stores it in memory.

	    Feel free to add instance variables to Model object that store loaded data.    
	    """
	    data_samples, labels = extract_imagedata(normalization=0)
	    aug_data = data_samples.reshape((data_samples.shape[0], data_samples.shape[1], data_samples.shape[2], 1))
	    self.X_train = aug_data[:self.config.num_train, :, :, :]
	    self.Y_train = labels[:self.config.num_train, :]
	    self.X_test = aug_data[(self.config.num_train + 1):, :, :, :]
	    self.Y_test = labels[(self.config.num_train + 1): ,  :]



	def add_placeholders(self):
	    """Adds placeholder variables to tensorflow computational graph.

	    Tensorflow uses placeholder variables to represent locations in a
	    computational graph where data is inserted.  These placeholders are used as
	    inputs by the rest of the model building code and will be fed data during
	    training.

	    See for more information:

	    https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
	    """
	    self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.DIM_PHI, self.config.DIM_ETA, self.config.num_channels))
	    self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.num_classes))
	    self.dropout_placeholder = tf.placeholder(tf.float32)

	def create_feed_dict(self, input_batch, dropout, label_batch=None):
	    """Creates the feed_dict for training the given step.

	    A feed_dict takes the form of:

	    feed_dict = {
	        <placeholder>: <tensor of values to be passed for placeholder>,
	        ....
	    }
	  
	    If label_batch is None, then no labels are added to feed_dict.

	    Hint: The keys for the feed_dict should be a subset of the placeholder
	          tensors created in add_placeholders.
	    
	    Args:
	      input_batch: A batch of input data.
	      label_batch: A batch of label data.
	    Returns:
	      feed_dict: The feed dictionary mapping from placeholders to values.
	    """
	    feed_dict = {
	      self.input_placeholder : input_batch,
	      self.dropout_placeholder : dropout
	    }
	    if label_batch is not None:
	    	feed_dict[self.labels_placeholder] = label_batch
	    return feed_dict

	def add_model(self, input_data):
		with tf.variable_scope("FirstConv") as CLayer1:
			w_conv1 = tf.get_variable("w_conv1", (11, 11, 1, 32), initializer=tf.truncated_normal_initializer(stddev=0.1))
			b_conv1 = tf.get_variable("b_conv1", (32), initializer=tf.constant_initializer(0.1))
			conv1 =   tf.nn.conv2d(input_data, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
			hconv1 =  tf.nn.relu(conv1 + b_conv1)
			h_pool1 = tf.nn.max_pool(hconv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			with tf.variable_scope("SecondConv") as CLayer2:
				w_conv2 = tf.get_variable("w_conv2", (11 , 11, 32, 64), initializer=tf.truncated_normal_initializer(stddev=0.1))
				b_conv2 = tf.get_variable("b_conv2", (64), initializer=tf.constant_initializer(0.1))
				conv2 =   tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
				hconv2 =  tf.nn.relu(conv2 + b_conv2)
				h_pool2 = tf.nn.max_pool(hconv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
				with tf.variable_scope("FullyConnected") as FC:
					wfc1 = tf.get_variable("wfc1", (self.config.final_size*64, 32), initializer=tf.truncated_normal_initializer(stddev=0.1))
					bfc1 = tf.get_variable("bfc1", (32), initializer=tf.constant_initializer(0.1))
					h_pool2_flat = tf.reshape(h_pool2, [-1, self.config.final_size*64])
					h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, wfc1) + bfc1)
					h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_placeholder)
					with tf.variable_scope("ReadoutLayer") as RL:
						wfc2 = tf.get_variable("wfc2", (32, self.config.num_classes), initializer=tf.truncated_normal_initializer(stddev=0.1))
						bfc2 = tf.get_variable("bfc2", (self.config.num_classes), initializer=tf.constant_initializer(0.1))
						y_conv = tf.matmul(h_fc1_drop, wfc2) + bfc2
		return y_conv

	def add_loss_op(self, pred):
	    """Adds ops for loss to the computational graph.

	    Args:
	      pred: A tensor of shape (batch_size, n_classes)
	    Returns:
	      loss: A 0-d tensor (scalar) output
	    """
	    #Hinge Loss
	    #pred = tf.nn.softmax(pred)
	    #loss = tf.reduce_mean(tf.maximum( 0.0, (1.0 - self.labels_placeholder)*pred - self.labels_placeholder*pred + 1.0 ))
	    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, self.labels_placeholder))
	    return loss

	def add_training_op(self, loss):
	    """Sets up the training Ops.

	    Creates an optimizer and applies the gradients to all trainable variables.
	    The Op returned by this function is what must be passed to the
	    `sess.run()` call to cause the model to train. See 

	    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

	    for more information.

	    Hint: Use tf.train.AdamOptimizer for this model.
	          Calling optimizer.minimize() will return a train_op object.

	    Args:
	      loss: Loss tensor, from cross_entropy_loss.
	    Returns:
	      train_op: The Op for training.
	    """
	    ### YOUR CODE HERE
	    optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
	    train_op = optimizer.minimize(loss)
	    return train_op

	def __init__(self, config):
	  	self.config = config
	  	self.load_data()
	  	self.add_placeholders()
	  	output = self.add_model(self.input_placeholder)
	  	self.loss = self.add_loss_op(output)
	  	self.predictions = tf.nn.softmax(output)
	  	one_hot_prediction = tf.argmax(self.predictions, 1)
	  	correct_prediction = tf.equal(tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
	  	self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
	  	self.train_op = self.add_training_op(self.loss)

	def run_epoch(self, sess, input_data, input_labels):

	    orig_X, orig_y = input_data, input_labels
	    dp = self.config.dropout
	    total_loss = []
	    total_correct_examples = 0
	    total_processed_examples = 0
	    total_steps = len(orig_X) / self.config.batch_size
	    for step, (x, y) in enumerate(data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
	                   label_size=self.config.num_classes)):

	        feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
	        loss, total_correct, _ = session.run(
	          [self.loss, self.correct_predictions, self.train_op],
	          feed_dict=feed)
	        total_processed_examples += len(x)
	        total_correct_examples += total_correct
	        total_loss.append(loss)
	        sys.stdout.write('\r{} / {} : loss = {}'.format(
		        step, total_steps, np.mean(total_loss)))
	        sys.stdout.flush()
	        sys.stdout.write('\r')
	        sys.stdout.flush()
	    return np.mean(total_loss), total_correct_examples / float(total_processed_examples)


	def predict(self, sess, input_data, input_classes):
	    """Make predictions from the provided model.
	    Args:
	      sess: tf.Session()
	      input_data: np.ndarray of shape (n_samples, n_features)
	      input_labels: np.ndarray of shape (n_samples, n_classes)
	    Returns:
	      average_loss: Average loss of model.
	      predictions: Predictions of model on input_data
	    """
	    dp = 1
	    losses = []
	    results = []
	    predictions_scores = []
	    data = data_iterator(input_data, batch_size=self.config.batch_size,
	                             label_size=self.config.num_classes)
	    first = True
	    for step, (x, y) in enumerate(data):
	      feed = self.create_feed_dict(input_batch=x, dropout=dp)
	      preds = session.run(self.predictions, feed_dict=feed)
	      preds = np.array(preds)
	      if first:
	      	predictions_scores = preds
	      	first = False
	      else:
	      	predictions_scores = np.vstack((predictions_scores, preds))
	      predicted_indices = preds.argmax(axis=1)
	      results.extend(predicted_indices)

	    flattened_classes = np.array(input_classes.argmax(axis=1))
	    accuracy = sum(np.array(results) == flattened_classes)/float(len(flattened_classes))
	    print 'Num vbf'
	    print np.sum(flattened_classes)
	    print 'Total objects'
	    print len(flattened_classes)
	    print 'Accuracy on Test'
	    print accuracy
	    indices = np.arange(flattened_classes.shape[0])
	    corresponding_scores = predictions_scores[indices, flattened_classes]
	    area_under_curve = roc_auc_score(flattened_classes, corresponding_scores)
	    print 'Here is the roc_auc_score 1'
	    print area_under_curve
	    print 'Mean loss'
	    print np.mean(losses)
	    print flattened_classes[0:5]
	    print corresponding_scores[0:5]
	    fpr, tpr , thresholds = roc_curve(flattened_classes, corresponding_scores)
	    plt.ylabel('True Positive Rate')
	    plt.xlabel('False Positive Rate')
	    area = auc(fpr, tpr)#np.trapz(efficiencies[:,1], x=efficiencies[:,0])
	    print 'This is the area'
	    print area
	    plt.plot(fpr, tpr)
	    title = "CNN Graph-CE-5050-16OUT"
	    plt.figtext(.4, .5, "AUC : " + str(area))
	    pp = PdfPages(title + ".pdf")
	    plt.savefig(pp, format="pdf")
	    pp.close()
	    plt.show()
	    return np.mean(losses), results


config = Config()
tf.reset_default_graph()
with tf.Graph().as_default():
	model = HEPModel(config)
	init = tf.initialize_all_variables()
	with tf.Session() as session:
		best_val_loss = float('inf')
      	best_val_epoch = 0
      	session.run(init)
      	for epoch in xrange(config.max_epoch):
      		print 'Epoch {}'.format(epoch)
     		train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                model.Y_train)
        	print 'Training loss: {}'.format(train_loss)
        	print 'Training acc: {}'.format(train_acc)
        	if train_loss < best_val_loss:
         	 	best_val_loss = train_loss
          		best_val_epoch = epoch
          	if epoch - best_val_epoch > config.early_stopping:
          		break	

    	val_loss, predictions = model.predict(session, model.X_test, model.Y_test)

