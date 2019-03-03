import numpy as np
import warnings
import pickle
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from scipy.special import expit as sigmoid
import sys

import tensorflow as tf
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

### Below base code for the ResNet50 model is taken from https://github.com/fchollet/deep-learning-models.git
### it has been modified to have YOLO classifier in the end layers (see ResNet50() function)
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
def ResNet50(include_top=False, load_weight=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected ResNet50 classifier
            layer at the top of the network or use the YOLO classifier
        load_weight: if True, load weights as specified in the 'weights' argument
        weights: when 'load_weight' is True, this specifies the path to model weights
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:

        ###------------- YOLO Classifier layer -----------###

        x = Flatten(name='yolo_clf_0')(x)
        x = Dense(2048, activation='relu', name='yolo_clf_1')(x)
        #x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.5, name='yolo_clf_2')(x)

        # output tensor :
        # SS: Grid cells: 11*11
        # B: Bounding box per grid cell: 2
        # C: classes: 3
        # Coords: x, y, w, h per box: 4
        # tensor length: SS * (C +B(5) ) : 363--242--968 => 1573
        x = Dense(11*11*(3+2*5), activation='linear', name='yolo_clf_3')(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50_yolo')

    # load weights
    if load_weight:
        if weights == 'imagenet':
            if include_top:
                weights_path = 'models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
            else:
                weights_path = 'models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        else:
            weights_path = weights
        # print(weights_path, '\n', save_prefix, '\n', learning_rate)
        # sys.exit()
        model.load_weights(weights_path, by_name=True)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')


    return model
#-----------------------------------------------------------------------#

#-----------------------------------------------------------------------#
def loop_body(t_true, t_pred, i, ta):
    '''
    This funtion is the main body of the custom_loss() definition, called from within the tf.while_loop()
    The loss funtion implemented here is as decsribed in the original YOLO paper: https://arxiv.org/abs/1506.02640

    # Arguments
    t_true: the ground truth tensor; shape: (batch_size, 1573)
    t_pred: the predicted tensor; shape: (batch_size, 1573)
    i: iteration cound of the while_loop
    ta: TensorArray that stores loss
    '''

    ### Get the current iteration's tru and predicted tensor
    c_true = t_true[i]
    c_pred = t_pred[i]
    ### Apply sigmoid to the coordinates part of the tensor to scale it between 0 and 1 as expected
    c_pred = tf.concat((c_pred[:605], tf.sigmoid(c_pred[-968:])), axis=0)

    ### Reshape to GRIDxGRIDxBBOXES blocks for simpler coorespondence of
    ### values across grid cell and bounding boxes
    xywh_true = tf.reshape(c_true[-968:], (11,11,2,4))
    xywh_pred = tf.reshape(c_pred[-968:], (11,11,2,4))

    ### Convert normalized values to actual ones (still relative to grid cell size)
    x_true = xywh_true[:,:,:,0] * X_NORM
    x_pred = xywh_pred[:,:,:,0] * X_NORM

    y_true = xywh_true[:,:,:,1] * Y_NORM
    y_pred = xywh_pred[:,:,:,1] * Y_NORM

    w_true = xywh_true[:,:,:,2] * WIDTH_NORM
    w_pred = xywh_pred[:,:,:,2] * WIDTH_NORM

    h_true = xywh_true[:,:,:,3] * HEIGHT_NORM
    h_pred = xywh_pred[:,:,:,3] * HEIGHT_NORM

    ### The below is a different approach on calculating IOU between
    ### predicted bounding boxes and ground truth
    ### See README.md for explanation for the formula
    x_dist = tf.abs(tf.subtract(x_true, x_pred))
    y_dist = tf.abs(tf.subtract(y_true, y_pred))

    ### (w1/2 +w2/2 -d) > 0 => intersection, else no intersection
    ### (h1/2 +h2/2 -d) > 0 => intersection, else no intersection
    wwd = tf.nn.relu(w_true/2 + w_pred/2 - x_dist)
    hhd = tf.nn.relu(h_true/2 + h_pred/2 - y_dist)

    area_true = tf.multiply(w_true, h_true)
    area_pred = tf.multiply(w_pred, h_pred)
    area_intersection = tf.multiply(wwd, hhd)

    iou = area_intersection / (area_true + area_pred - area_intersection + 1e-4)
    confidence_true = tf.reshape(iou, (-1,))

    ### Masks for grids that do contain an object, from ground truth
    ### The class probability block from the ground truth is used as an indicator for all grid cells that
    ### actually have an object present in itself.
    grid_true = tf.reshape(c_true[:363], (11,11,3))
    grid_true_sum = tf.reduce_sum(grid_true, axis=2)
    grid_true_exp = tf.stack((grid_true_sum, grid_true_sum), axis=2)
    grid_true_exp3 = tf.stack((grid_true_sum, grid_true_sum, grid_true_sum), axis=2)
    grid_true_exp4 = tf.stack((grid_true_sum, grid_true_sum, grid_true_sum, grid_true_sum), axis=2)

    coord_mask = tf.reshape(grid_true_exp4, (-1,))
    confidence_mask = tf.reshape(grid_true_exp, (-1,))
    confidence_true = confidence_true * confidence_mask


    ### Revised ground truth tensor, based on calculated confidence values and with non-object grids suppressed
    c_true_new = tf.concat([c_true[:363], confidence_true, c_true[-968:]], axis=0)

    ### Create masks for 'responsible' bounding box in a grid cell for loss calculation
    confidence_true_matrix = tf.reshape(confidence_true, (11,11,2))
    confidence_true_argmax = tf.argmax(confidence_true_matrix, axis=2)
    confidence_true_argmax = tf.cast(confidence_true_argmax, tf.int32)
    ind_i, ind_j = tf.meshgrid(tf.range(11), tf.range(11), indexing='ij')
    ind_argmax = tf.stack((ind_i, ind_j, confidence_true_argmax), axis=2)
    ind_argmax = tf.reshape(ind_argmax, (121,3))

    responsible_mask_2 = tf.scatter_nd(ind_argmax, tf.ones((121)), [11,11,2])
    responsible_mask_2 = tf.reshape(responsible_mask_2, (-1,))
    responsible_mask_2 = responsible_mask_2 * confidence_mask

    responsible_mask_4 = tf.scatter_nd(ind_argmax, tf.ones((121,2)), [11,11,2,2])
    responsible_mask_4 = tf.reshape(responsible_mask_4, (-1,))
    responsible_mask_4 = responsible_mask_4 * coord_mask

    ### Masks for rest of the bounding boxes
    inv_responsible_mask_2 = tf.cast(tf.logical_not(tf.cast(responsible_mask_2, tf.bool)), tf.float32)
    inv_responsible_mask_4 = tf.cast(tf.logical_not(tf.cast(responsible_mask_4, tf.bool)), tf.float32)

    ### lambda values
    lambda_coord = 5.0
    lambda_noobj = 0.5

    ### loss from dimensions ###
    dims_true = tf.reshape(c_true_new[-968:], (11,11,2,4))
    dims_pred = tf.reshape(c_pred[-968:], (11,11,2,4))

    xy_true = tf.reshape(dims_true[:,:,:,:2], (-1,))
    xy_pred = tf.reshape(dims_pred[:,:,:,:2], (-1,))

    wh_true = tf.reshape(dims_true[:,:,:,2:], (-1,))
    wh_pred = tf.reshape(dims_pred[:,:,:,2:], (-1,))

    #### XY difference loss
    xy_loss = (xy_true - xy_pred) * responsible_mask_4
    xy_loss = tf.square(xy_loss)
    xy_loss = lambda_coord * tf.reduce_sum(xy_loss)


    #### WH sqrt diff loss
    wh_loss = (tf.sqrt(wh_true) - tf.sqrt(tf.abs(wh_pred))) * responsible_mask_4
    wh_loss = tf.square(wh_loss)
    wh_loss = lambda_coord * tf.reduce_sum(wh_loss)


    ### Conf losses
    conf_true = c_true_new[363:605]
    conf_pred = c_pred[363:605]

    conf_loss_obj = (conf_true - conf_pred) * responsible_mask_2
    conf_loss_obj = tf.square(conf_loss_obj)
    conf_loss_obj = tf.reduce_sum(conf_loss_obj)


    conf_loss_noobj = (conf_true - conf_pred) * inv_responsible_mask_2
    conf_loss_noobj = tf.square(conf_loss_noobj)
    conf_loss_noobj = lambda_noobj * tf.reduce_sum(conf_loss_noobj)


    #### Class Prediction Loss
    class_true = tf.reshape(c_true_new[:363], (11,11,3))
    class_pred = tf.reshape(c_pred[:363], (11,11,3))
    class_pred_softmax = class_pred #tf.nn.softmax(class_pred)

    classification_loss = class_true - class_pred_softmax
    classification_loss = classification_loss * grid_true_exp3
    classification_loss = tf.square(classification_loss)
    classification_loss = tf.reduce_sum(classification_loss)


    ## Total loss = xy-loss + wh-loss + Confidence_loss_obj + Confidence_loss_noobj + classification_loss
    total_loss = xy_loss + wh_loss + conf_loss_obj + conf_loss_noobj + classification_loss

    #debug
    #ta_debug = ta_debug.write(0, total_loss)
    #ta_debug = ta_debug.write(1, xy_loss)
    #ta_debug = ta_debug.write(2, wh_loss)
    #ta_debug = ta_debug.write(3, conf_loss_obj)
    #ta_debug = ta_debug.write(4, conf_loss_noobj)
    #ta_debug = ta_debug.write(5, classification_loss)

    ta = ta.write(i, total_loss)
    i = i+1
    return t_true, t_pred, i, ta

def custom_loss(y_true, y_pred):
    '''
    custom loss function as per the YOLO paper, since there are no default
    loss functions in TF or Keras that fit
    '''
    c = lambda t, p, i, ta : tf.less(i, tf.shape(t)[0])
    ta = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
    #ta_debug = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

    ### tf.while_loop creates a Tensorflow map with our loss function calculation (in loop_body())
    t, p, i, ta = tf.while_loop(c, loop_body, [y_true, y_pred, 0, ta])

    ### convert TensorArray into a tensor and calculate mean loss
    loss_tensor = ta.stack()
    loss_mean = tf.reduce_mean(loss_tensor)

    return loss_mean #, ta_debug.pack()
#-----------------------------------------------------------------------#

### Helper funtions for data augumentation for training the network ###
def coord_translate(bboxes, tr_x, tr_y):
    '''
    Takes a single frame's bounding box list with confidence scores and
    applies translation (addition) to the coordinates specified by 'tr'

    parameters:
    bboxes: list with element of the form ((x1,y1), (x2,y2)), (c1,c2,c3)
    tr_x, tr_y: translation factor to add the coordinates to, for x and y respectively

    returns: new list with translated coordinates and same conf scores; same shape as bboxes
    '''
    new_list = []
    for box in bboxes:
        coords = np.array(box[0])
        coords[:,0] = coords[:,0] + tr_x
        coords[:,1] = coords[:,1] + tr_y
        coords = coords.astype(np.int64)
        out_of_bound_indices = np.average(coords, axis=0) >= 224
        if out_of_bound_indices.any():
            continue
        coords = coords.tolist()
        new_list.append((coords, box[1]))
    return new_list
def coord_scale(bboxes, sc):
    '''
    Takes a singl frame's bounding box list with confidence scores and
    applies scaling to the coordinates specified by sc

    parameters:
    bboxes: list with element of the form ((x1,y1), (x2,y2)), (c1,c2,c3)
    sc: scaling factor to multiply the coordinates with

    returns: new list with scaled coordinates and same conf scores; same shape as bboxes
    '''
    new_list = []
    for box in bboxes:
        coords = np.array(box[0])
        coords = coords * sc
        coords = coords.astype(np.int64)
        out_of_bound_indices = np.average(coords, axis=0) >= 224
        if out_of_bound_indices.any():
            continue
        coords = coords.tolist()
        new_list.append((coords, box[1]))
    return new_list
def label_to_tensor(frame, imgsize=(224, 224), gridsize=(11,11), classes=3, bboxes=2):
    '''
    This function takes in the frame (rows corresponding to a single image in the labels.csv)
    and converts it into the format our network expects (coord conversion and normalization)

    '''
    grid = np.zeros(gridsize)

    y_span = imgsize[0]/gridsize[0]
    x_span = imgsize[1]/gridsize[1]

    class_prob = np.zeros((gridsize[0], gridsize[1], classes))
    confidence = np.zeros((gridsize[0], gridsize[1], bboxes))
    dims = np.zeros((gridsize[0], gridsize[1], bboxes, 4))

    for box in frame:
        ((x1,y1), (x2,y2)), (c1,c2,c3) = box
        x_grid = int(((x1+x2)/2)//x_span)
        y_grid = int(((y1+y2)/2)//y_span)

        class_prob[y_grid, x_grid] = (c1,c2,c3)

        x_center = ((x1+x2)/2)
        y_center = ((y1+y2)/2)

        x_center_norm = (x_center-x_grid*x_span)/(x_span)
        y_center_norm = (y_center-y_grid*y_span)/(y_span)

        w = x2-x1
        h = y2-y1

        w_norm = w/imgsize[1]
        h_norm = h/imgsize[0]

        dims[y_grid, x_grid, :, :] = (x_center_norm, y_center_norm, w_norm, h_norm)

        grid[y_grid, x_grid] += 1

    tensor = np.concatenate((class_prob.ravel(), confidence.ravel(), dims.ravel()))
    return tensor
def augument_data(label, frame, imgsize=(224, 224), folder='udacity-object-detection-crowdai/'):
    '''
    Takes the image file name and the frame (rows corresponding to a single image in the labels.csv)
    and randomly scales, translates, adjusts SV values in HSV space for the image,
    and adjusts the coordinates in the 'frame' accordingly, to match bounding boxes in the new image
    '''
    img = cv2.imread(folder+label)
    img = cv2.resize(img, imgsize)
    rows, cols = img.shape[:2]

    #translate_factor
    tr = np.random.random() * 0.2 + 0.01
    tr_y = np.random.randint(rows*-tr, rows*tr)
    tr_x = np.random.randint(cols*-tr, cols*tr)
    #scale_factor
    sc = np.random.random() * 0.4 + 0.8

    # flip coin to adjust image saturation
    r = np.random.rand()
    if r < 0.5:
        #randomly adjust the S and V values in HSV representation
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        fs = np.random.random() + 0.7
        fv = np.random.random() + 0.2
        img[:,:,1] *= fs
        img[:,:,2] *= fv
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # new random factor for scaling and translating
    r = np.random.rand()

    if r < 0.3:
        #translate image
        M = np.float32([[1,0,tr_x], [0,1,tr_y]])
        img = cv2.warpAffine(img, M, (cols,rows))
        frame = coord_translate(frame, tr_x, tr_y)
    elif r < 0.6:
        #scale image keeping the same size
        placeholder = np.zeros_like(img)
        meta = cv2.resize(img, (0,0), fx=sc, fy=sc)
        if sc < 1:
            placeholder[:meta.shape[0], :meta.shape[1]] = meta
        else:
            placeholder = meta[:placeholder.shape[0], :placeholder.shape[1]]
        img = placeholder
        frame = coord_scale(frame, sc)

    return img, frame
#-----------------------------------------------------------------------#

### Define generator and Import dataset (do test/train split)
def generator(label_keys, label_frames, batch_size=64, folder='udacity-object-detection-crowdai/'):
    '''
    Generator function
    # Arguments
    label_keys: image names, that are keys of the label_frames Arguments
    label_frames: array of frames (rows corresponding to a single image in the labels.csv)
    batch_size: batch size
    '''
    num_samples = len(label_keys)
    indx = label_keys

    while 1:
        shuffle(indx)
        for offset in range(0, num_samples, batch_size):
            batch_samples = indx[offset:offset+batch_size]

            images = []
            gt = []
            for batch_sample in batch_samples:
                im, frame = augument_data(batch_sample, label_frames[batch_sample])
                im = im.astype(np.float32)
                im -= 128
                images.append(im)
                frame_tensor = label_to_tensor(frame)
                gt.append(frame_tensor)

            X_train = np.array(images)
            y_train = np.array(gt)
            yield shuffle(X_train, y_train)

def plot_history(history_object):
    print(history_object.history.keys())
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    ### Constants for our model:
    WIDTH_NORM = 224
    HEIGHT_NORM = 224
    GRID_NUM = 11
    X_SPAN = WIDTH_NORM/GRID_NUM
    Y_SPAN = HEIGHT_NORM/GRID_NUM
    X_NORM = WIDTH_NORM/GRID_NUM
    Y_NORM = HEIGHT_NORM/GRID_NUM

    weights_path = 'imagenet'
    save_prefix = 'run1_'
    learning_rate = 1e-2
    if len(sys.argv) > 3:
        weights_path = sys.argv[1]
        save_prefix = sys.argv[2]
        learning_rate = float(sys.argv[3])
    elif len(sys.argv) > 2:
        weights_path = sys.argv[1]
        save_prefix = sys.argv[2]
    elif len(sys.argv) > 1:
        weights_path = sys.argv[1]

    model = ResNet50(include_top=False, input_shape=(224,224,3),
                    load_weight=True, weights=weights_path)

    with open('label_frames.p', 'rb') as f:
        label_frames = pickle.load(f)

    label_keys = list(label_frames.keys())
    lbl_train, lbl_validn = train_test_split(label_keys, test_size=0.2)

    ### Intialize generator
    train_generator = generator(lbl_train, label_frames)
    validation_generator = generator(lbl_validn, label_frames)

    ### Compile model
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss=custom_loss)
    model_checkpoint = ModelCheckpoint(filepath='models/' + save_prefix + str(learning_rate) + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, mode='auto', save_weights_only=True, period=1, verbose=1)
    history = model.fit_generator(train_generator, validation_data=validation_generator,
                                    steps_per_epoch=len(lbl_train)//64, epochs=15,
                                    validation_steps=len(lbl_validn)//64,
                                    callbacks=[model_checkpoint])
    model.save_weights('models/'+save_prefix+str(learning_rate))
