from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Activation, UpSampling2D
from keras.optimizers import SGD
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import add

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

    return model


def get_unet_256(input_shape=(256, 256, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

    return model


def get_unet_512(input_shape=(512, 512, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='classify')(up0a)

    model = Model(inputs=inputs, outputs=[classify, classify])

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=[dice_loss])

    return model

def block0(in_layer, nchan, relu=True):
    b1 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(in_layer)
    # b1 = BatchNormalization()(b1)
    if relu:
        b1 = Activation('relu')(b1)
    else:
        b1 = LeakyReLU(0.0001)(b1)

    b2 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b1)
    # b2 = BatchNormalization()(b2)
    if relu:
        b2 = Activation('relu')(b2)
    else:
        b2 = LeakyReLU(0.0001)(b2)

    # b3 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b2)
    # # b3 = BatchNormalization()(b3)
    # if relu:
    #     b3 = Activation('relu')(b3)
    # else:
    #     b3 = LeakyReLU(0.0001)(b3)
    #
    # b4 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b3)
    # # b4 = BatchNormalization()(b4)
    # if relu:
    #     b4 = Activation('relu')(b4)
    # else:
    #     b4 = LeakyReLU(0.0001)(b4)
    #
    # out_layer = concatenate([b1, b4], axis=3)
    # out_layer = Conv2D(nchan, (1, 1), padding='same')(out_layer)
    # if relu:
    #     out_layer = Activation('relu')(out_layer)
    # else:
    #     out_layer = LeakyReLU(0.0001)(out_layer)
    return b2

def block(in_layer, nchan, relu=True):
    b1 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(in_layer)
    # b1 = BatchNormalization()(b1)
    if relu:
        b1 = Activation('relu')(b1)
    else:
        b1 = LeakyReLU(0.0001)(b1)

    b2 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b1)
    # b2 = BatchNormalization()(b2)
    if relu:
        b2 = Activation('relu')(b2)
    else:
        b2 = LeakyReLU(0.0001)(b2)

    b3 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b2)
    # b3 = BatchNormalization()(b3)
    if relu:
        b3 = Activation('relu')(b3)
    else:
        b3 = LeakyReLU(0.0001)(b3)

    b4 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b3)
    # b4 = BatchNormalization()(b4)
    if relu:
        b4 = Activation('relu')(b4)
    else:
        b4 = LeakyReLU(0.0001)(b4)

    out_layer = concatenate([b1, b4], axis=3)
    out_layer = Conv2D(nchan, (1, 1), padding='same')(out_layer)
    if relu:
        out_layer = Activation('relu')(out_layer)
    else:
        out_layer = LeakyReLU(0.0001)(out_layer)
    return out_layer

def block1(in_layer, nchan, relu=True):
    m = nchan // 2
    b1 = Conv2D(m, (3, 3), padding='same')(in_layer)
    # b1 = BatchNormalization()(b1)
    if relu:
        b1 = Activation('relu')(b1)
    else:
        b1 = LeakyReLU(0.0001)(b1)

    b2 = Conv2D(m, (3, 3), padding='same')(b1)
    # b2 = BatchNormalization()(b2)
    if relu:
        b2 = Activation('relu')(b2)
    else:
        b2 = LeakyReLU(0.0001)(b2)

    b3 = Conv2D(m, (3, 3), padding='same')(b2)
    # b3 = BatchNormalization()(b3)
    if relu:
        b3 = Activation('relu')(b3)
    else:
        b3 = LeakyReLU(0.0001)(b3)


    b5 = Conv2D(m, (3, 3), padding='same')(in_layer)
    b5 = Activation('relu')(b5)


    b6 = Conv2D(m, (1, 1), padding='same')(in_layer)
    b6 = Activation('relu')(b6)

    b7 = Conv2D(m, (3, 3), padding='same')(in_layer)
    b7 = Activation('relu')(b7)
    b7 = Conv2D(m, (3, 3), padding='same')(b7)
    b7 = Activation('relu')(b7)

    # out_layer = add([b1, b5])
    out_layer = concatenate([b3, b5, b6, b7], axis=3)
    out_layer = Conv2D(nchan, (1, 1), padding='same')(out_layer)
    return out_layer

def block2(in_layer, nchan, relu=True):
    b0 = Conv2D(nchan, (3, 3), padding='same')(in_layer)
    # b1 = BatchNormalization()(b1)
    if relu:
        b1 = Activation('relu')(b0)
    else:
        b1 = LeakyReLU(0.0001)(b0)

    b2 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b1)
    # b2 = BatchNormalization()(b2)
    if relu:
        b2 = Activation('relu')(b2)
    else:
        b2 = LeakyReLU(0.0001)(b2)

    b3 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b2)
    # b3 = BatchNormalization()(b3)
    if relu:
        b3 = Activation('relu')(b3)
    else:
        b3 = LeakyReLU(0.0001)(b3)

    b4 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b3)
    # b4 = BatchNormalization()(b4)
    if relu:
        b4 = Activation('relu')(b4)
    else:
        b4 = LeakyReLU(0.0001)(b4)

    b5 = Conv2D(nchan, (3, 3), padding='same', kernel_initializer='he_uniform')(b4)
    # b4 = BatchNormalization()(b4)

    out_layer = add([b0, b5])
    if relu:
        out_layer = Activation('relu')(out_layer)
    else:
        out_layer = LeakyReLU(0.0001)(out_layer)
    #out_layer = Conv2D(nchan, (1, 1), padding='same')(out_layer)
    return out_layer

def block3(in_layer, nchan):
    b1b = Conv2D(nchan, (1, 3), padding='same', kernel_initializer='he_uniform')(in_layer)
    b1b = Activation('relu')(b1b)
    b1b = Conv2D(nchan, (3, 1), padding='same', kernel_initializer='he_uniform')(b1b)
    b1b = Activation('relu')(b1b)

    b1a = Conv2D(nchan, (3, 1), padding='same', kernel_initializer='he_uniform')(in_layer)
    b1a = Activation('relu')(b1a)
    b1a = Conv2D(nchan, (1, 3), padding='same', kernel_initializer='he_uniform')(b1a)
    b1a = Activation('relu')(b1a)

    #b1 = concatenate([b1b, b1a], axis=3)
    b1 = add([b1b, b1a])
    b1 = Conv2D(nchan, (1, 1), padding='same')(b1)
    b2b = Conv2D(nchan, (1, 3), padding='same', kernel_initializer='he_uniform')(b1)
    b2b = Activation('relu')(b2b)
    b2b = Conv2D(nchan, (3, 1), padding='same', kernel_initializer='he_uniform')(b2b)
    b2b = Activation('relu')(b2b)

    b2a = Conv2D(nchan, (3, 1), padding='same', kernel_initializer='he_uniform')(b1)
    b2a = Activation('relu')(b2a)
    b2a = Conv2D(nchan, (1, 3), padding='same', kernel_initializer='he_uniform')(b2a)
    b2a = Activation('relu')(b2a)

    #b2 = concatenate([b2b, b2a], axis=3)
    b2 = add([b2b, b2a])
    b2 = Conv2D(nchan, (1, 1), padding='same')(b2)
    b3b = Conv2D(nchan, (1, 3), padding='same')(b2)
    b3b = Activation('relu')(b3b)
    b3b = Conv2D(nchan, (3, 1), padding='same')(b3b)
    b3b = Activation('relu')(b3b)

    b3a = Conv2D(nchan, (3, 1), padding='same')(b2)
    b3a = Activation('relu')(b3a)
    b3a = Conv2D(nchan, (1, 3), padding='same')(b3a)
    b3a = Activation('relu')(b3a)
    out_layer = add([b1, b3a, b3b])
    #out_layer = concatenate([b3b, b3a], axis=3)
    #out_layer = Conv2D(nchan, (1, 1), padding='same')(b3)
    #out_layer = Conv2D(nchan, (1, 1), padding='same')(out_layer)
    return out_layer

def get_unet_1024(input_shape=(1024, 1024, 3),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 1024
    down0b = block0(inputs, 16)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)

    # 512
    down0a = block0(down0b_pool, 32)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    # 256
    down0 = block0(down0a_pool, 64)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    # 128
    down1 = block0(down0_pool, 128)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    # 64
    down2 = block0(down1_pool, 256)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    # 32
    down3 = block0(down2_pool, 512)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    # 16
    down4 = block0(down3_pool, 1024)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

    # 8
    # center = block(down4_pool, 1024)
    dilate1 = Conv2D(1024, (3, 3), activation='relu', padding='same', dilation_rate=1)(down4_pool)
    dilate2 = Conv2D(1024, (3, 3), activation='relu', padding='same', dilation_rate=2)(dilate1)
    dilate3 = Conv2D(1024, (3, 3), activation='relu', padding='same', dilation_rate=4)(dilate2)
    dilate4 = Conv2D(1024, (3, 3), activation='relu', padding='same', dilation_rate=8)(dilate3)
    dilate5 = Conv2D(1024, (3, 3), activation='relu', padding='same', dilation_rate=16)(dilate4)
    dilate6 = Conv2D(1024, (3, 3), activation='relu', padding='same', dilation_rate=32)(dilate5)
    center = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])

    # center
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = block0(up4, 1024)

    # 16
    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = block0(up3, 512)

    # 32
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = block0(up2, 256)

    # 64
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = block0(up1, 128)

    # 128
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = block0(up0, 64)

    # 256
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = block0(up0a, 32)

    # 512
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = block0(up0b, 16)

    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    return model

def get_unet_512_nobn(input_shape=(512, 512, 3),
                  num_classes=1):
    inputs = Input(shape=input_shape)
    # 512
    down0b = block(inputs, 32)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)

    # 256
    down0a = block(down0b_pool, 64)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)

    # 128
    down0 = block(down0a_pool, 128)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)

    # 64
    down1 = block(down0_pool, 256)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

    # 32
    down2 = block(down1_pool, 512)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    # 16
    down3 = block(down2_pool, 1024)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

    # 8
    center = block(down3_pool, 1024)

    # center
    up3 = UpSampling2D((2, 2))(center)
    up3 = concatenate([down3, up3], axis=3)
    up3 = block(up3, 1024)

    # 16
    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = block(up2, 512)

    # 32
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = block(up1, 256)

    # 64
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = block(up0, 128)

    # 128
    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = block(up0a, 64)

    # 256
    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = block(up0b, 32)

    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss, metrics=[dice_loss])
    return model
