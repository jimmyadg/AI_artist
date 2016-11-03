from scipy.misc import imread, imresize, imsave
from scipy.optimize import fmin_l_bfgs_b
from sklearn.preprocessing import normalize
import numpy as np
import time
import os
import argparse
import h5py

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D
from keras import backend as K

#base image, style reference, result ouput
args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix

#getting weights file
weights_path = r"vgg16_weights.h5"

#resizing if needed
rescale_image = strToBool(args.rescale_image)
maintain_aspect_ratio = strToBool(args.maintain_aspect_ratio)

#style & content weights
total_variation_weight = args.tv_weight
style_weight = args.style_weight * args.style_scale
content_weight = args.content_weight

#dimensions of generated pic
img_width = img_height = args.img_size
assert img_height == img_width, 'width and height must match'
img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0

#tensor representation of original image
base_image = K.variable(preprocess_image(base_image_path, True))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

#generated image
combination_image = K.placeholder((1,3,img_width,img_height))

#combine 3 imgs into a simgle keras tensor
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

#passing the 3 imgs in to build a VGG16 network
first_layer = ZeroPadding2D((1,1))
first_layer.set_input(input_tensor, shape=(3,3,img_width,img_height))

#tensor layers (rectified linear unit)
model = Sequential()
model.add(first_layer)
model.add(Convolution2D(64,3,3,activation='relu',name='conv1_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(AveragePooling2D((2,2),strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3,activation='relu',name = 'conv2_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128,3,3,activation='relu'))
model.add(AveragePooling2D((2,2),strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3,activation='relu',name='conv3_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256,3,3,activation='relu'))
model.add(AveragePooling2D((2,2),strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,activation='relu',name='conv4_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,activation='relu',name='conv4_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,activation='relu'))
model.add(AveragePooling2D((2,2),strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,activation='relu',name='conv5_1'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,activation='relu',name='conv5_2'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512,3,3,activation='relu'))
model.add(AveragePooling2D((2,2),strides=(2,2)))

#load the weights of VGG16 networks
load_weights(weights_path,model)

#get the symbolic outputs of each key layer
outputs_dict = dict([(layer.ouput) for layer in model.layers])

#get the loss
loss = get_total_loss(outputs_dict)

#get the gradients of the generated image ith the loss
grads = K.gradients(loss,combination_image)

#combine loss and gradients
f_outputs = combine_loss_and_gradient(loss,grads)

#script optimization
x,num_iter - prepare_image()
for i in range(num_iter):
    #record iterations
    print('Start of iterations',(i+1))
    start_time = time.time()

    #perform l_bfgs optimizer using loss and gard
    x,min_val,info = fmin_l_bfgs_b(evaluator.loss,x.flatten(),
                                   fprime=evaluator.grads,maxfun=20)
    print('Current loss value: ', min_val)

    #get the generated img
    img = deprocess_image(x.reshape((3,img_width,img_height)))

    #maitain ascpect ratio
    if(maintain_aspect_ratio) & (not rescale_image):
        img_ht = int(img_width * aspect_ratio)
        print("Rescaling Image to (%d,%d)" % (img_width,img_ht))
        img = innersize(img, (img_width,img_ht),interp=args.rescale_method)
    if rescale_image:
        print("Rescaling image to (%d.%d)" % (img_WIDTH,img_HEIGHT))
        img = imresize(img,(img_WIDTH,img_HEIGHT),interp=args.rescale_method)

    #save generated img
    fname = result_prefix + '_at_iteration_%d.png' % (i+1)
    imsave(fname,img)
