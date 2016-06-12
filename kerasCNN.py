import numpy as np 
from extract_data_from_trees import extract_imagedata
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt 
import sys
import h5py

#If you need to import function from file in another folder.
# import sys
# sys.path.insert(0, 'path')

from matplotlib.backends.backend_pdf import PdfPages
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.layers.core import Reshape
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import Adam


DIM_ETA = 52
DIM_PHI = 64
TEST_FRAC = 0.1
#path where weights of the model can be saved.
SAVED_WEIGHTS_PATH = 'saved_path/weights_noht.hdf5'

#MODEL Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 20
DROP_OUT = 0.1
REGULARIZATION = 1e-3
EARLY_STOPPING = 2
CONV_FILTER_SIZE = 5
CONV_MODE = 'valid'
LEARNING_RATE = 1e-4
INITIALIZATION = "glorot_normal"

train_fracs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
performances = np.zeros(len(train_fracs));
index = 0;
#Get the data

data_samples, labels, htsoft = extract_imagedata(normalization=0)
#use the code below to save and load data so you don't have to extract the data from the tree and reprocess it every time
# np.save("save_path/samples", data_samples)
# np.save("save_path/labels", labels)
# np.save("save_path/ht", htsoft)
# data_samples = np.load("save_path/samples.npy")
# labels = np.load("save_path/labels.npy")
# ht = np.load("saved_path/ht.npy")

data_samples = data_samples.reshape((data_samples.shape[0], 1,  data_samples.shape[1], data_samples.shape[2]))
num_samples = data_samples.shape[0]
num_test_start = int(np.floor((1.0 - TEST_FRAC)*num_samples))
x_test = data_samples[num_test_start:, :, :]
y_test = labels[num_test_start:, :]
for frac in train_fracs:
	#Get the training set
	num_train = int(np.floor(frac*num_samples))
	x_train = data_samples[:num_train, :, :]
	y_train = labels[:num_train, :]

	#Build the residual model
	model = Sequential()
	model.add(Convolution2D(32, CONV_FILTER_SIZE, CONV_FILTER_SIZE, init=INITIALIZATION,  border_mode=CONV_MODE, input_shape=(1,DIM_PHI, DIM_ETA), W_regularizer=l2(REGULARIZATION)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
	model.add(Convolution2D(64, CONV_FILTER_SIZE, CONV_FILTER_SIZE, init=INITIALIZATION,  border_mode=CONV_MODE, W_regularizer=l2(REGULARIZATION)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
	model.add(Flatten())
	model.add(Dense(32, activation='relu', init=INITIALIZATION,  W_regularizer=l2(REGULARIZATION))) # returns a 32 dimensional representation of each image sample 

	inputDependence = Sequential()
	inputDependence.add(Flatten(input_shape=(1, DIM_PHI, DIM_ETA)))
	inputDependence.add(Dense(32, activation='relu', init=INITIALIZATION, W_regularizer=l2(REGULARIZATION))) # 

	merged_model = Sequential()
	merged = Merge([model, inputDependence], mode='sum')
	merged_model.add(merged)
	merged_model.add(Dropout(DROP_OUT))
	merged_model.add(Activation('relu'))
	merged_model.add(Dense(2, activation="softmax", init=INITIALIZATION, W_regularizer=l2(REGULARIZATION)))

	optim = Adam(lr=LEARNING_RATE)
	merged_model.compile(loss="categorical_crossentropy", optimizer=optim,  metrics=['accuracy'])
	early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING, verbose=0, mode='min')
	checkpointer = ModelCheckpoint(SAVED_WEIGHTS_PATH, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
	merged_model.fit([x_train, x_train], y_train, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_split=0.3, callbacks=[early_stopper, checkpointer])
	
	#Evaluate the model
	merged_model.load_weights(SAVED_WEIGHTS_PATH)
	score = merged_model.evaluate([x_test, x_test], y_test, batch_size=BATCH_SIZE)
	print 'Test Accuracy'
	print score[1]
	#get the prediction probabilities
	predictions_scores = merged_model.predict_proba([x_test, x_test], batch_size=x_test.shape[0])
	actual_classes = np.array(y_test.argmax(axis=1)).reshape(len(y_test), 1) #converts matrix of labels to vector of 0 (ggf) and 1(vbf)
	indices = np.arange(actual_classes.shape[0]).reshape(actual_classes.shape[0], 1)
	corresponding_scores = predictions_scores[:, 1]
	fpr, tpr , thresholds = roc_curve(actual_classes, corresponding_scores)
	print 'Test Area Under Curve'
	area = auc(fpr, tpr)
	print area
	performances[index] = area
	index += 1
	plt.plot(fpr, tpr)
	title = "CNNGraph-" + str(frac) 
	plt.figtext(.4, .5, "AUC : " + str(area))
	pp = PdfPages(title + ".pdf")
	plt.savefig(pp, format="pdf")
	pp.close()
	plt.close()
plt.ylabel('AUC SCORE')
plt.xlabel('FRACTION OF DATA FOR TRAIN')
plt.plot(train_fracs, performances)
title = "DATA VRS PERFORMANCE"
pp = PdfPages(title + ".pdf")
plt.savefig(pp, format="pdf")
pp.close()
plt.show()
	
