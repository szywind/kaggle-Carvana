# -*- coding: utf-8 -*-
__author__ = 'Zhenyuan Shen: https://kaggle.com/szywind'

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, time, gc, imutils, cv2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from keras import optimizers

# from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.model_selection import KFold

from helpers import *
import newnet
import math
import glob
import random
from PIL import Image

np.set_printoptions(threshold='nan')

INPUT_PATH = '../input/'
OUTPUT_PATH = '../output/'

class CarvanaCarSeg():
    def __init__(self, input_dim=512, batch_size=16, epochs=10, learn_rate=1e-4, nb_classes=2):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.nb_classes = nb_classes
        self.model = newnet.fcn_32s(input_dim, nb_classes)
        self.nAug = 2 # incl. horizon mirror augmentation
        # self.nTTA = 1 # incl. horizon mirror augmentation
        self.load_data()

    def load_data(self):
        self.train_masks = np.array(glob.glob(INPUT_PATH + 'train_masks/*.gif'))
        self.train_imgs = np.array(glob.glob(INPUT_PATH + 'train/*.jpg'))
        self.test_imgs = np.array(glob.glob(INPUT_PATH + 'test/*.jpg'))
        index = list(range(len(self.train_imgs)))
        random.shuffle(index)
        self.train_masks = self.train_masks[index]
        self.train_masks = self.train_masks[index]

    def train(self):

        # train_datagen = ImageDataGenerator(
        #     rescale=1. / 255,
        #     zoom_range=0.15,
        #     rotation_range=360,
        #     width_shift_range=0.1,
        #     height_shift_range=0.1
        # )
        # val_datagen = ImageDataGenerator(rescale=1. / 255)

        num_fold = 0

        thres = []
        val_score = 0




        # train_datagen.fit(x_train, augment=True, rounds=2, seed=1)
        # train_generator = train_datagen.flow(x_train[train_index], y_train[train_index], shuffle=True, batch_size=batch_size, seed=int(time.time()))
        # val_generator = val_datagen.flow(x_train[test_index], y_train[test_index], shuffle=False, batch_size=batch_size)

        nTrain = len(self.train_imgs)

        print('Training on {} samples'.format(nTrain))


        train_x = self.train_imgs
        train_y = self.train_masks

        def train_generator():
            while True:
                for start in range(0, nTrain, self.batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + self.batch_size, nTrain)

                    for i in range(start, end):
                        j = np.random.randint(self.nAug)
                        img = cv2.imread(train_x[i])
                        img = cv2.resize(img, (self.input_dim, self.input_dim), interpolation=cv2.INTER_LINEAR)
                        img = transformations2(img, j)
                        x_batch.append(img)

                        mask = np.array(Image.open(train_y[i]), dtype=np.uint8)
                        mask = cv2.resize(mask, (self.input_dim, self.input_dim), interpolation=cv2.INTER_LINEAR)
                        mask = transformations2(mask, j)
                        target = np.zeros((mask.shape[0], mask.shape[1], self.nb_classes))
                        for k in range(self.nb_classes):
                            target[:,:,k] = (mask == k)
                        y_batch.append(target)

                    x_batch = np.array(x_batch, np.float32)
                    y_batch = np.array(y_batch, np.float32)
                    yield x_batch, y_batch

        model_path = os.path.join('', 'car-segmentation-model.h5')

        opt  = optimizers.SGD(lr=self.learn_rate, momentum=0.9)
        self.model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                          optimizer=opt,
                          metrics=['accuracy'])

        callbacks = [ModelCheckpoint(model_path, save_best_only=False, verbose=0)]



        self.model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=math.ceil(nTrain / float(self.batch_size)),
            callbacks=callbacks,
            epochs=self.epochs,
            verbose=2)


        opt  = optimizers.SGD(lr=0.1*self.learn_rate)
        self.model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                          optimizer=opt,
                          metrics=['accuracy'])

        self.model.fit_generator(
            generator=train_generator(),
            steps_per_epoch=math.ceil(nTrain / float(self.batch_size)),
            callbacks=callbacks,
            epochs=self.epochs,
            verbose=2)

    def test(self):
        nTest = len(self.test_imgs)
        print('Testing on {} samples'.format(nTest))

        def test_generator(transformation):
            while True:
                for start in range(0, nTest, self.batch_size):
                    x_batch = []
                    end = min(start + self.batch_size, nTest)

                    for i in range(start, end):
                        img = cv2.imread(self.test_imgs[i])
                        img = cv2.resize(img, (self.input_dim, self.input_dim), interpolation=cv2.INTER_LINEAR)
                        img = transformations2(img, transformation)
                        x_batch.append(img)
                    x_batch = np.array(x_batch, np.float32)
                    yield x_batch


        model_path = os.path.join('', 'car-segmentation-model.h5')
        if not os.path.isfile(model_path):
            raise RuntimeError("No model found.")

        self.model.load_weights(model_path)

        print('Create submission...')
        t = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
        str = []
        nbatch = 0
        for start in range(0, nTest, self.batch_size):
            print(nbatch)
            nbatch += 1
            x_batch = []
            end = min(start + self.batch_size, nTest)
            for i in range(start, end):
                img = cv2.imread(self.test_imgs[i])
                img = cv2.resize(img, (self.input_dim, self.input_dim), interpolation=cv2.INTER_LINEAR)
                img = transformations2(img, 0)
                x_batch.append(img)
            x_batch = np.array(x_batch, np.float32)

            p_test = self.model.predict(x_batch, batch_size=self.batch_size)

            avg_p_test = p_test[...,1] - p_test[...,0]
            result = get_result(avg_p_test, 0)

            str.extend(map(rle, result))
            # save predicted masks
            if not os.path.exists(OUTPUT_PATH):
                os.mkdir(OUTPUT_PATH)

            for i in range(len(result)):
                cv2.imwrite(OUTPUT_PATH + '/{}'.format(
                    self.test_imgs[start+i][self.test_imgs[start+i].rfind('/')+1:]), (255 * result[i]).astype(np.uint8))

        t['rle_mask'] = str
        t.to_csv('subm_{}.gz'.format(100), index=False, compression='gzip')

    # def create_submission(self, best_score, predict_masks):
    #     print('Create submission...')
    #     t = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    #     str = map(rle, predict_masks)
    #     t['rle_mask'] = str
    #     t.to_csv('subm_{}.gz'.format(best_score), index=False, compression='gzip')

if __name__ == "__main__":
    ccs = CarvanaCarSeg()

    ccs.train()
    ccs.test()

    # af.refine(thresh, val_score)
