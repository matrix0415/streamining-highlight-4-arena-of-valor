# The code was referenced from https://github.com/wohlert/keras-squeezenet/.
from uuid import uuid4
import keras.backend as K
from keras.models import Model
from keras.utils.data_utils import get_file
from keras.layers import Input, Dropout, Concatenate, Activation, Conv2D, \
    MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dense, Lambda, Reshape
from keras.applications.imagenet_utils import _obtain_input_shape

WEIGHTS_PATH = 'https://github.com/wohlert/keras-squeezenet/releases/download/v0.1/squeezenet_weights.h5'


def _fire(x, filters, name):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Conv2D(sq_filters, (1, 1), activation='relu', padding='same', name=name + "/squeeze1x1")(x)
    expand1 = Conv2D(ex1_filters, (1, 1), activation='relu', padding='same', name=name + "/expand1x1")(squeeze)
    expand2 = Conv2D(ex2_filters, (3, 3), activation='relu', padding='same', name=name + "/expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name + '/concatenate')([expand1, expand2])
    return x


def _fc_layer(prev_layer, classes):
    x = Dropout(0.5)(prev_layer)
    x = Conv2D(classes, (1, 1), padding='valid')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    return x


def SqueezeNet(include_top=True, weights=None, input_shape=None, pooling=None, classes=1000, input_tensor=None):
    u = str(uuid4())[-7:]
    # Determine proper input shape
    if input_tensor is None:
        input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=48, data_format=K.image_data_format(), require_flatten=False)
        input_tensor = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding="same", activation="relu", name=u+'/conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name=u+'/maxpool1', padding="valid")(x)
    x = _fire(x, (16, 64, 64), name=u+"/fire2")
    x = _fire(x, (16, 64, 64), name=u+"/fire3")
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=u+'/maxpool3', padding="valid")(x)
    x = _fire(x, (32, 128, 128), name=u+"/fire4")
    x = _fire(x, (32, 128, 128), name=u+"/fire5")
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=u+'/maxpool5', padding="valid")(x)
    x = _fire(x, (48, 192, 192), name=u+"/fire6")
    x = _fire(x, (48, 192, 192), name=u+"/fire7")
    x = _fire(x, (64, 256, 256), name=u+"/fire8")
    x = _fire(x, (64, 256, 256), name=u+"/fire9")

    x = _fc_layer(prev_layer=x, classes=1000)

    if pooling == "avg":
        x = GlobalAveragePooling2D(name=u+"/avgpool10")(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name=u+"/maxpool10")(x)

    model = Model(input_tensor, x, name=u+"/squeezenet")

    if weights == 'imagenet':
        weights_path = get_file('squeezenet_weights.h5', WEIGHTS_PATH, cache_subdir='models')
        print("Load weights.......")
        model.load_weights(weights_path)

        if classes != 1000:
            x = model.layers[-5].output
            # x = _fc_layer(prev_layer=x, classes=classes)
            x = BatchNormalization()(x)
            x = GlobalAveragePooling2D()(x)
            x = Dense(classes, input_dim=512, activation='sigmoid', name=u+'/fc/sigmoid')(x)
            # x = Conv2D(classes, (1, 1), padding='valid')(x)
            # x = Activation('softmax')(x)
            model = Model(input_tensor, x, name=u+"/squeezenet")

    if not include_top:
        x = model.layers[-5].output
        model = Model(input_tensor, x, name=u + "/squeezenet")

    return model
