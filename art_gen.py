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
K.set_image_dim_ordering('th')


####cl argumentst#####

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('--base_image_path', metavar='base', type=str, help='Path to the image to transform.')
parser.add_argument('--style_reference_image_path', metavar='ref', type=str, help='Path to the style reference image.')
parser.add_argument('--result_prefix', metavar='res_prefix', type=str, help='Prefix for the saved results.')

parser.add_argument("--image_size", dest="img_size", default=512, type=int, help='Output Image size')
parser.add_argument("--content_weight", dest="content_weight", default=0.025, type=float, help="Weight of content") # 0.025
parser.add_argument("--style_weight", dest="style_weight", default=1, type=float, help="Weight of content") # 1.0
parser.add_argument("--style_scale", dest="style_scale", default=1.0, type=float, help="Scale the weightage of the style") # 1, 0.5, 2
parser.add_argument("--total_variation_weight", dest="tv_weight", default=1e-3, type=float, help="Total Variation in the Weights") # 1.0
parser.add_argument("--num_iter", dest="num_iter", default=10, type=int, help="Number of iterations")
parser.add_argument("--rescale_image", dest="rescale_image", default="True", type=str, help="Rescale image after execution to original dimentions")
parser.add_argument("--rescale_method", dest="rescale_method", default="bilinear", type=str, help="Rescale image algorithm")
parser.add_argument("--maintain_aspect_ratio", dest="maintain_aspect_ratio", default="True", type=str, help="Maintain aspect ratio of image")
parser.add_argument("--content_layer", dest="content_layer", default="conv5_2", type=str, help="Optional 'conv4_2'")
parser.add_argument("--init_image", dest="init_image", default="content", type=str, help="Initial image used to generate the final image. Options are 'content' or 'noise")


####function wrapper by llSourcell####
def strToBool(v):
    return v.lower() in ("true", "yes", "t", "1")

# adjusting piture to fit tensor
def preprocess_image(image_path, load_dims=False):
    global img_WIDTH, img_HEIGHT, aspect_ratio

    img = imread(image_path) # Prevents crashes due to PNG images (ARGB)
    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

    img = imresize(img, (img_width, img_height))
    img = img.transpose((2, 0, 1)).astype('float64')
    img = np.expand_dims(img, axis=0)
    return img

#converting tensor into a valid image
def deprocess_image(x):
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_weights(weight_path, model):
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')


#image tensor gram matrix
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram

def eval_loss_and_grads(x):
    x = x.reshape((1, 3, img_width, img_height))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

#feature mapping of the reference picture
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, 1:, :img_height-1])
    b = K.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, :img_width-1, 1:])
    return K.sum(K.pow(a + b, 1.25))

def get_total_loss(outputs_dict):
    # combine loss funcs
    loss = K.variable(0.)
    layer_features = outputs_dict[args.content_layer] # 'conv5_2' or 'conv4_2'
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features,
                                      combination_features)
    feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

def combine_loss_and_gradient(loss, gradient):
    outputs = [loss]
    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)
    f_outputs = K.function([combination_image], outputs)
    return f_outputs

def prepare_image():
    assert args.init_image in ["content", "noise"] , "init_image must be one of ['original', 'noise']"
    if "content" in args.init_image:
        x = preprocess_image(base_image_path, True)
    else:
        x = np.random.uniform(0, 255, (1, 3, img_width, img_height))
    num_iter = args.num_iter
    return x, num_iter

# scipy.optimize loss and grads
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()



#variables
args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix

weights_path = r"vgg16_weights.h5" #weight file

#resizing determining boolean
rescale_image = strToBool(args.rescale_image)
maintain_aspect_ratio = strToBool(args.maintain_aspect_ratio)

# style and content weight
total_variation_weight = args.tv_weight
style_weight = args.style_weight * args.style_scale
content_weight = args.content_weight

# demension of the generate image
img_width = img_height = args.img_size
assert img_height == img_width, 'Due to the use of the Gram matrix, width and height must match.'
img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0

#tensor
base_image = K.variable(preprocess_image(base_image_path, True))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

combination_image = K.placeholder((1, 3, img_width, img_height))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

# using the three inputs to build a VGG network
first_layer = ZeroPadding2D((1, 1))
first_layer.set_input(input_tensor, shape=(3, 3, img_width, img_height))

model = Sequential()
model.add(first_layer)
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

#####Code by Somshubra Majumdar#####
# load the weights of the VGG16 networks
load_weights(weights_path, model)

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# get the loss (we combine style, content, and total variation loss into a single scalar)
loss = get_total_loss(outputs_dict)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

#combine loss and gradient
f_outputs = combine_loss_and_gradient(loss, grads)

# Run scipy-based optimization (L-BFGS) over the pixels of the generated image to minimize the neural style loss
# 5 Step process
x, num_iter = prepare_image()
for i in range(num_iter):

    #Step 1 - Record iterations
    print('Start of iteration', (i+1))
    start_time = time.time()

    #Step 2 - Perform l_bfgs optimization function using loss and gradient
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)

    #Step 3 - Get the generated image
    img = deprocess_image(x.reshape((3, img_width, img_height)))

    #Step 4 - Maintain aspect ratio
    if (maintain_aspect_ratio) & (not rescale_image):
        img_ht = int(img_width * aspect_ratio)
        print("Rescaling Image to (%d, %d)" % (img_width, img_ht))
        img = imresize(img, (img_width, img_ht), interp=args.rescale_method)
    if rescale_image:
        print("Rescaling Image to (%d, %d)" % (img_WIDTH, img_HEIGHT))
        img = imresize(img, (img_WIDTH, img_HEIGHT), interp=args.rescale_method)

    #Step 5 - Save the generated image
    fname = 'iteration_%d.png' % (i+1)
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i+1, end_time - start_time))
