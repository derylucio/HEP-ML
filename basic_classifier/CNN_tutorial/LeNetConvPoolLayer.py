import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy as np
from theano.tensor.signal import downsample

class LeNetConvPoolLayer(object):
	"""Pool Layer of a convolutional network """
	def __init__(self, rng, input, filter_shape, img_shape, pool_size=(2,2)):
		        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        assert img_shae[1] == filter_shape[1] 
        self.input = input
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = filter_shape[0]*np.prod(filter_shape[2:])/ np.prod(pool_size)
        W_bound = np.sqrt(6.0/(fan_in + fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=dtype=theano.config.floatX), borrow=True)
        b_values = np.zeros((filter_shape[0],)dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        conv_out = conv2d(input=input, filters=self.W, filter_shape=filter_shape, input_shape=img_shape)
        pool_out = downsample.max_pool_2d(input=conv_out, ds=pool_size, ignore_border=True)
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input


