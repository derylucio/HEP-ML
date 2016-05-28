import numpy as np 
from extract_data_from_trees import extract_imagedata
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt 
import sys
from matplotlib.backends.backend_pdf import PdfPages
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.layers.core import Reshape
import keras.callbacks
from keras.regularizers import l2
from keras.optimizers import Adam

DIM_ETA = 52
DIM_PHI = 64
TRAIN_FRAC = 0.9
BATCH_SIZE = 128
NUM_EPOCHS = 20
REGULARIZATION = 1e-3

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.poor = []
        self.first = True
        self.losses.append(logs.get('loss'))

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


data_samples, labels, htsoft = extract_imagedata(normalization=1)
data_samples = data_samples.reshape((data_samples.shape[0], 1,  data_samples.shape[1], data_samples.shape[2]))
num_train = int(np.floor(TRAIN_FRAC*data_samples.shape[0]))

x_train = data_samples[:num_train, :, :]
y_train = labels[:num_train, :]
x_test = data_samples[num_train:, :, :]
y_test = labels[num_train:, :]

model = Sequential()
model.add(Convolution2D(32, 11, 11, border_mode="same", input_shape=(1,DIM_PHI, DIM_ETA), W_regularizer=l2(REGULARIZATION)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64, 11, 11, border_mode="same", W_regularizer=l2(REGULARIZATION)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32, W_regularizer=l2(REGULARIZATION)))

inputDependence = Sequential()
inputDependence.add(Flatten(input_shape=(1, DIM_PHI, DIM_ETA)))
inputDependence.add(Dense(32))
merged_model = Sequential()
merged = Merge([model, inputDependence], mode='sum')
merged_model.add(merged)
merged_model.add(Dropout(0.1))
merged_model.add(Activation('relu'))
merged_model.add(Dense(2, W_regularizer=l2(REGULARIZATION)))



optim = Adam(lr=1e-4)
merged_model.compile(loss="categorical_crossentropy", optimizer=optim,  metrics=['accuracy'])
history = LossHistory()
merged_model.fit([x_train, x_train], y_train, callbacks=[history], batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS, validation_split=0.3)
#print history.losses
print history.losses
score = merged_model.evaluate([x_test, x_test], y_test, batch_size=BATCH_SIZE)

predictions_scores = merged_model.predict_proba([x_test, x_test], batch_size=x_test.shape[0])
print predictions_scores.shape
flattened_classes = np.array(y_test.argmax(axis=1))
indices = np.arange(flattened_classes.shape[0])
corresponding_scores = predictions_scores[indices, flattened_classes]
print 'Here is the test performance'
print score
fpr, tpr , thresholds = roc_curve(flattened_classes, corresponding_scores)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
area = auc(fpr, tpr)
print area
plt.plot(fpr, tpr)
title = "CNNGraph-CE-Ydep-LargeDataMean-Keras"
plt.figtext(.4, .5, "AUC : " + str(area))
pp = PdfPages(title + ".pdf")
plt.savefig(pp, format="pdf")
pp.close()