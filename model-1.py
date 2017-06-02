#==================================================================================================
# Classic model
#
# X -> [CONV->CONV->POOL]x5 -> DENSE -> DENSE -> Y
#==================================================================================================
import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import rectify, sigmoid, linear, tanh
from lasagne.layers import InputLayer, DenseLayer, BatchNormLayer, \
                           Upscale2DLayer, NonlinearityLayer, ReshapeLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, dropout
#==================================================================================================
np.random.seed(12)
#==================================================================================================
# CONST
#--------------------------------------------------------------------------------------------------
second_dim = 1
img_size = 160
blocks_count = 5
#--------------------------------------------------------------------------------------------------
# TODO: load it from file
#--------------------------------------------------------------------------------------------------
conv_filter_size = [0 for _ in range(blocks_count)]
conv_num_filters = [0 for _ in range(blocks_count)]
conv_pad =    [0 for _ in range(blocks_count)]
conv_stride = [0 for _ in range(blocks_count)]
pool_size =   [0 for _ in range(blocks_count)]
#--------------------------------------------------------------------------------------------------
conv_filter_size[0] = 3
conv_num_filters[0] = 64
conv_pad[0]         = 1
conv_stride[0]      = 1
#--------------------------------------------------------------------------------------------------
conv_filter_size[1] = 3
conv_num_filters[1] = 128
conv_pad[1]         = 1
conv_stride[1]      = 1
#--------------------------------------------------------------------------------------------------
conv_filter_size[2] = 3
conv_num_filters[2] = 256
conv_stride[2]      = 1
conv_pad[2]         = 1
#--------------------------------------------------------------------------------------------------
conv_filter_size[3] = 3
conv_num_filters[3] = 512
conv_stride[3]      = 1
conv_pad[3]         = 1
#--------------------------------------------------------------------------------------------------
conv_filter_size[4] = 3
conv_num_filters[4] = 512
conv_stride[4]      = 1
conv_pad[4]         = 1
#--------------------------------------------------------------------------------------------------
pool_size[0] = 2
pool_size[1] = 2
pool_size[2] = 2
pool_size[3] = 2
pool_size[4] = 2
#--------------------------------------------------------------------------------------------------
dense_layer1_size = 1024
dense_layer1_drop = 0.5
dense_layer2_size = 512
dense_layer2_drop = 0.5
#==================================================================================================
my_nonlin = rectify
#==================================================================================================
input_image_left     = T.tensor4('input_left')
input_image_positive = T.tensor4('input_positive')
input_image_negative = T.tensor4('input_negative')
#==================================================================================================
l_input = InputLayer(shape=(None, second_dim, img_size, img_size), input_var=input_image_left)
p_input = InputLayer(shape=(None, second_dim, img_size, img_size), input_var=input_image_positive)
n_input = InputLayer(shape=(None, second_dim, img_size, img_size), input_var=input_image_negative)
#==================================================================================================
# Creating blocks [CONV->CONV->POOL] for LEFT
#--------------------------------------------------------------------------------------------------
net = l_input
for i in range(blocks_count):
    net = Conv2DLayer(net, conv_num_filters[i], conv_filter_size[i], pad=conv_pad[i],
                      stride=conv_stride[i], nonlinearity=my_nonlin)
    net = Conv2DLayer(net, conv_num_filters[i], conv_filter_size[i], pad=conv_pad[i],
                      stride=conv_stride[i], nonlinearity=my_nonlin)
    net = MaxPool2DLayer(net, pool_size[i])
#--------------------------------------------------------------------------------------------------
net = DenseLayer(dropout(net, p=dense_layer1_drop), num_units=dense_layer1_size, 
                 nonlinearity=my_nonlin)
nn_l_out = DenseLayer(dropout(net, p=dense_layer2_drop), num_units=dense_layer2_size, 
                      nonlinearity=my_nonlin)
#--------------------------------------------------------------------------------------------------
l_params = lasagne.layers.get_all_params(nn_l_out)
#==================================================================================================
# Creating blocks [CONV->CONV->POOL] for POSITIVE
#--------------------------------------------------------------------------------------------------
net = p_input
for i in range(blocks_count):
    net = Conv2DLayer(net, conv_num_filters[i], conv_filter_size[i], pad=conv_pad[i],
                      stride=conv_stride[i], nonlinearity=my_nonlin,
                      W=l_params[4*i], b=l_params[4*i+1])
    net = Conv2DLayer(net, conv_num_filters[i], conv_filter_size[i], pad=conv_pad[i], 
                      stride=conv_stride[i], nonlinearity=my_nonlin,
                      W=l_params[4*i+2], b=l_params[4*i+3])
    net = MaxPool2DLayer(net, pool_size[i])
#--------------------------------------------------------------------------------------------------
net = DenseLayer(dropout(net, p=dense_layer1_drop), num_units=dense_layer1_size, 
                 nonlinearity=my_nonlin,
                 W=l_params[blocks_count*4], b=l_params[blocks_count*4+1])
nn_p_out = DenseLayer(dropout(net, p=dense_layer2_drop), num_units=dense_layer2_size,
                 nonlinearity=my_nonlin,
                 W=l_params[blocks_count*4+2], b=l_params[blocks_count*4+3])
#==================================================================================================
# Creating blocks [CONV->CONV->POOL] for NEGATIVE
#--------------------------------------------------------------------------------------------------
net = n_input
for i in range(blocks_count):
    net = Conv2DLayer(net, conv_num_filters[i], conv_filter_size[i], pad=conv_pad[i],
                      stride=conv_stride[i], nonlinearity=my_nonlin,
                      W=l_params[4*i], b=l_params[4*i+1])
    net = Conv2DLayer(net, conv_num_filters[i], conv_filter_size[i], pad=conv_pad[i], 
                      stride=conv_stride[i], nonlinearity=my_nonlin,
                      W=l_params[4*i+2], b=l_params[4*i+3])
    net = MaxPool2DLayer(net, pool_size[i])
#--------------------------------------------------------------------------------------------------
net = DenseLayer(dropout(net, p=dense_layer1_drop), num_units=dense_layer1_size, 
                 nonlinearity=my_nonlin,
                 W=l_params[blocks_count*4], b=l_params[blocks_count*4+1])
nn_n_out = DenseLayer(dropout(net, p=dense_layer2_drop), num_units=dense_layer2_size,
                 nonlinearity=my_nonlin,
                 W=l_params[blocks_count*4+2], b=l_params[blocks_count*4+3])
#==================================================================================================
nn_merge = lasagne.layers.concat([nn_l_out, nn_p_out, nn_n_out], axis=1)
nn_out      = lasagne.layers.get_output(nn_merge, deterministic=False)
nn_out_test = lasagne.layers.get_output(nn_merge, deterministic=True)
nn_out_spec = lasagne.layers.get_output(nn_l_out, deterministic=True) # Get vector from image
#--------------------------------------------------------------------------------------------------
nn_out_left     = nn_out[:, :dense_layer2_size]
nn_out_positive = nn_out[:, dense_layer2_size:dense_layer1_size]
nn_out_negative = nn_out[:, dense_layer1_size:]
#--------------------------------------------------------------------------------------------------
nn_out_left_test     = nn_out_test[:, :dense_layer2_size]
nn_out_positive_test = nn_out_test[:, dense_layer2_size:dense_layer1_size]
nn_out_negative_test = nn_out_test[:, dense_layer1_size:]
#--------------------------------------------------------------------------------------------------
a = T.scalar()
#--------------------------------------------------------------------------------------------------
d1   = T.sum(T.sqr(nn_out_left - nn_out_positive), axis=1)
d2   = T.sum(T.sqr(nn_out_left - nn_out_negative), axis=1)
loss = T.sum(T.maximum(T.sqr(d1) - T.sqr(d2) + a, 0.))
#--------------------------------------------------------------------------------------------------
d1_test   = T.sum(T.sqr(nn_out_left_test - nn_out_positive_test), axis=1)
d2_test   = T.sum(T.sqr(nn_out_left_test - nn_out_negative_test), axis=1)
test_loss = T.sum(T.maximum(T.sqr(d1_test) - T.sqr(d2_test) + a, 0.))
#--------------------------------------------------------------------------------------------------
params = lasagne.layers.get_all_params(nn_merge)
updates = lasagne.updates.adamax(loss, params)
#==================================================================================================
train_fn = theano.function([input_image_left, input_image_positive, input_image_negative, a], 
                           loss, 
                           updates=updates, allow_input_downcast=True)
#==================================================================================================
val_fn = theano.function([input_image_left, input_image_positive, input_image_negative, a], 
                         test_loss, 
                         updates=updates, allow_input_downcast=True)
#==================================================================================================
test_fn = theano.function([input_image_left, input_image_positive, input_image_negative],
                          [d1_test, d2_test], 
                          allow_input_downcast=True)
#==================================================================================================
get_vec = theano.function([input_image_left], nn_out_spec, allow_input_downcast=True)
#==================================================================================================
