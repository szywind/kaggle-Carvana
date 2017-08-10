from keras.models import Model
from keras.layers import Input, concatenate, ZeroPadding2D, Conv2D, MaxPooling2D, AveragePooling2D, \
    BatchNormalization, Activation, UpSampling2D
from keras.initializers import he_uniform
from keras.layers.merge import add, Concatenate
from keras.layers.core import Dropout

def unit_1(in_layer, n1=64, n2=64, n3=256, s2=1, p2=1, d2=1):
    '''
    Two-Brach Unit
    :param in_layer:
    :return:
    '''
    # branch 1
    x1 = Conv2D(n1, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(in_layer)
    x1 = BatchNormalization(momentum=0.95)(x1)
    x1 = Activation('relu')(x1)

    x1 = ZeroPadding2D(padding=(p2, p2))(x1)
    x1 = Conv2D(n2, (3, 3), strides=(s2, s2), padding='valid', dilation_rate=(d2, d2), kernel_initializer=he_uniform(), use_bias=False)(x1)
    x1 = BatchNormalization(momentum=0.95)(x1)
    x1 = Activation('relu')(x1)

    x1 = Conv2D(n3, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(x1)
    x1 = BatchNormalization(momentum=0.95)(x1)

    # branch 2
    x2 = Conv2D(n3, (1, 1), strides=(s2, s2), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(in_layer)
    x2 = BatchNormalization(momentum=0.95)(x2)

    x = add([x1, x2])
    x = Activation('relu')(x)
    return x

def unit_2(in_layer, n1=64, n2=64, n3=256, p2=1, d2=1):
    '''
    Shortcut Unit
    :param in_layer:
    :return:
    '''
    x = Conv2D(n1, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(in_layer)
    x = BatchNormalization(momentum=0.95)(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(p2, p2))(x)
    x = Conv2D(n2, (3, 3), strides=(1, 1), padding='valid', dilation_rate=(d2, d2), kernel_initializer=he_uniform(), use_bias=False)(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Activation('relu')(x)

    x = Conv2D(n3, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(x)
    x = BatchNormalization(momentum=0.95)(x)

    x = add([in_layer, x])
    x = Activation('relu')(x)
    return x

def unit_3(in_layer):
    '''
    Pyramid Pooling
    :param in_layer:
    :return:
    '''
    def pyramid(pool_size, stride):
        x = AveragePooling2D(pool_size=pool_size, strides=stride, padding='valid')(in_layer)
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(x)
        x = UpSampling2D(stride)(x)
        return x

    x1 = pyramid(60, 60)
    x2 = pyramid(30, 30)
    x3 = pyramid(20, 20)
    x4 = pyramid(10, 10)
    return Concatenate([in_layer, x1, x2, x3, x4])

def pspnet(input_shape=(473, 473, 3), num_classes=1):
    inputs = Input(shape=input_shape)
    # 237
    x = ZeroPadding2D()(inputs)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', kernel_initializer=he_uniform(), use_bias=False)(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=he_uniform(), use_bias=False)(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=he_uniform(), use_bias=False)(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Activation('relu')(x)

    # 119
    x = ZeroPadding2D()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    x = unit_1(x, n1=64, n2=64, n3=256, s2=1)
    for _ in range(2):
        x = unit_2(x, n1=64, n2=64, n3=256)

    # 60
    x = unit_1(x, n1=128, n2=128, n3=512, s2=2)
    for _ in range(3):
        x = unit_2(x, n1=128, n2=128, n3=512)

    # conv3_4
    x = unit_1(x, n1=256, n2=256, n3=1024, s2=1, p2=2, d2=2)
    for _ in range(22):
        x = unit_2(x, n1=256, n2=256, n3=1024, s2=1, p2=2, d2=2)

    # conv4_23
    x = unit_1(x, n1=512, n2=512, n3=2048, s2=1, p2=4, d2=4)
    for _ in range(2):
        x = unit_2(x, n1=512, n2=512, n3=2048, s2=1, p2=4, d2=4)

    # conv5_3
    x = unit_3(x) # return x is 60x60x4096

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=he_uniform(), use_bias=False)(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)  # return x is 60x60x512

    x = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='same', kernel_initializer=he_uniform(), use_bias=False)(x)
    classify = UpSampling2D(8)(x)

    model = Model(inputs=inputs, outputs=classify)

    return model
