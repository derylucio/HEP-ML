import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import numpy as np
import pylab
from PIL import Image

rng = np.random.RandomState(23455)
input = T.tensor4(name="input")
#initialize the weight matrix
w_shape = (2, 3, 9, 9)
w_bound = np.sqrt(3*9*9)
W = theano.shared(np.asarray(rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shape),
            dtype=input.dtype), name="W")
b_shape = (2,)
bias = theano.shared(np.asarray(rng.uniform(low=-0.5, high=0.5, size=b_shape), dtype=input.dtype), name="b")
conv_out = conv2d(input, W)
output = T.nnet.sigmoid(conv_out + bias.dimshuffle('x', 0, 'x', 'x'))
#create a thiano function to compute filtered images 
f = theano.function([input], output)

#play around with this
img = Image.open(open('/Users/luciodery/Desktop/Kylo.jpg'))
img = np.asarray(img, dtype='float64') / 256.0
print img.shape
img_ = img.transpose(2, 0, 1).reshape(1, img.shape[2], img.shape[0], img.shape[1])
filtered_img = f(img_)

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
print 'Here'
pylab.show()

