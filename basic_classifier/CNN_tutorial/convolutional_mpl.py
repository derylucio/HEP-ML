import theano  
from theano import tensor as T
import numpy as np 
from LeNetConvPoolLayer import LeNetConvPoolLayer
from LogisticRegression import LogisticRegression
from HiddenLayer import HiddenLayer
import sys
import math
sys.path.insert(0, "/Users/luciodery/Desktop/Projects/Research/HEP+ML")
from extract_data_from_trees import extract_imagedata


def update_size(x_lim, y_lim, filter_size, poolsize):
	x_lim = (x_lim  - filter_size + 1)/poolsize
	y_lim = (y_lim  - filter_size + 1)/poolsize
	return x_lim, y_lim

batch_size = 30;
DIM_ETA = 50
DIM_PHI = 63
filter_sizes = [5, 5, 5]
nkerns = [4, 4, 3]
num_hidden = 500
num_classes = 2
split = np.array([0.6, 0.2, 0.2])

data_samples, labels = extract_imagedata()
data_samples = data_samples.reshape(data_samples.shape[0], 1, data_samples.shape[1], data_samples.shape[2])
num_split = np.floor(split*data_samples.shape[0])
train_set_x = data_samples[0:num_split[0], :, :,:]
train_set_y = labels[0:num_split[0], :, :,:]
valid_set_x = data_samples[num_split[0]:(num_split[0] + num_split[1]), :,:,:]
valid_set_y = labels[num_split[0]:(num_split[0] + num_split[1]), :,:,:]
test_set_x 	= data_samples[(num_split[0] + num_split[1]):, :,:, :]
test_set_y 	= labels[(num_split[0] + num_split[1]):, :,:]

x = T.matrix('x') 
y = T.ivector('y')
curr_x_dim = DIM_PHI
curr_y_dim = DIM_ETA
layer0_input = x.reshape((batch_size, 1, curr_x_dim, curr_y_dim))
layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, curr_x_dim, curr_y_dim),
        filter_shape=(nkerns[0], 1, filter_sizes[0], filter_sizes[0]),
        poolsize=(2, 2)
)

curr_x_dim, curr_y_dim = update_size(curr_x_dim, curr_y_dim, filter_sizes[0], 2)
print layer0.output.shape 
print curr_x_dim, curr_y_dim

layer1 = LeNetConvPoolLayer(
    rng,
    input= layer0.output,
    image_shape=(batch_size, nkerns[0], curr_x_dim, curr_y_dim),
    filter_shape=(nkerns[1], nkerns[0], filter_sizes[1], filter_sizes[1]),
    poolsize=(2, 2)
)

layer2_input = layer1.output.flatten(2)

curr_x_dim, curr_y_dim = update_size(curr_x_dim, curr_y_dim, filter_sizes[1], 2)
print layer0.output.shape 
print curr_x_dim, curr_y_dim
# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(
    rng,
    input=layer2_input,
    n_in=nkerns[1] * curr_x_dim * curr_y_dim,
    n_out=num_hidden,
    activation=T.tanh
)

layer3 = LogisticRegression(input=layer2.output, n_in=num_hidden, n_out=2)
cost = layer3.negative_log_likelihood(y)

test_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: test_set_x[index * batch_size: (index + 1) * batch_size],
        y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    [index],
    layer3.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.
updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        x: train_set_x[index * batch_size: (index + 1) * batch_size],
        y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

