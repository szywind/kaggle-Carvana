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

INPUT_PATH = '../input/'
OUTPUT_PATH = './'

class CarvanaCarSeg():
    def __init__(self, input_dim=512, batch_size=256, nfolds=5, epochs=300, learn_rate=1e-4):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.nfolds = nfolds
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.model = newnet.model3(input_dim, 2)
        self.nAug = 2 # incl. horizon mirror augmentation
        self.nTTA = 2 # incl. horizon mirror augmentation
        self.load_data()

    def load_data(self):
        self.train_masks = np.array(glob.glob(INPUT_PATH + 'train_masks/*.gif'))
        self.train_imgs = np.array(glob.glob(INPUT_PATH + 'train/*.jpg'))
        self.test_imgs = np.array(glob.glob(INPUT_PATH + 'test/*.jpg'))
        # index = list(range(len(self.imgs)))
        # random.shuffle(index)
        # self.imgs = self.imgs[index]
        # self.masks = self.masks[index]

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
        # kf = KFold(len(self.df_train_data), n_folds=self.nfolds, shuffle=True, random_state=1)  # deprecated KFold
        kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=1)

        for train_index, test_index in kf.split(self.train_imgs):
            num_fold += 1

            # train_datagen.fit(x_train, augment=True, rounds=2, seed=1)
            # train_generator = train_datagen.flow(x_train[train_index], y_train[train_index], shuffle=True, batch_size=batch_size, seed=int(time.time()))
            # val_generator = val_datagen.flow(x_train[test_index], y_train[test_index], shuffle=False, batch_size=batch_size)

            train_x = self.train_imgs[train_index]
            train_y = self.train_masks[train_index]
            valid_x = self.train_imgs[test_index]
            valid_y = self.train_masks[test_index]

            nTrain = len(train_x)
            nValid = len(valid_x)

            print('Training on {} samples'.format(nTrain))
            print('Validating on {} samples'.format(nValid))

            def valid_generator():
                while True:
                    for start in range(0, nValid, self.batch_size):
                        x_batch = []
                        y_batch = []
                        end = min(start + self.batch_size, nValid)

                        for i in range(start, end):
                            # j = np.random.randint(self.nAug)
                            img = cv2.imread(valid_x[i])
                            img = cv2.resize(img, (self.input_dim, self.input_dim))
                            # img = transformations2(img, j)
                            x_batch.append(img)

                            mask = np.array(Image.open(valid_y[i]), dtype=np.uint8) # mask = cv2.imread(valid_y[i]) cannot read .gif
                            mask = cv2.resize(mask, (self.input_dim, self.input_dim))
                            # mask = transformations2(mask, j)
                            y_batch.append(mask)
                        x_batch = np.array(x_batch, np.float32)
                        y_batch = np.array(y_batch, np.float32)
                        yield x_batch, y_batch

            def train_generator():
                while True:
                    for start in range(0, nTrain, self.batch_size):
                        x_batch = []
                        y_batch = []
                        end = min(start + self.batch_size, nTrain)

                        for i in range(start, end):
                            j = np.random.randint(self.nAug)
                            img = cv2.imread(train_x[i])
                            img = cv2.resize(img, (self.input_dim, self.input_dim))
                            img = transformations2(img, j)
                            x_batch.append(img)

                            mask = np.array(Image.open(train_y[i]), dtype=np.uint8)
                            mask = cv2.resize(mask, (self.input_dim, self.input_dim))
                            mask = transformations2(mask, j)
                            y_batch.append(mask)
                        x_batch = np.array(x_batch, np.float32)
                        y_batch = np.array(y_batch, np.float32)
                        yield x_batch, y_batch

            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

            opt  = optimizers.Adam(lr=self.learn_rate)
            self.model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                          optimizer=opt,
                          metrics=['accuracy'])

            # callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6),
            #              EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0),
            #              ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

            callbacks = [EarlyStopping(monitor='val_loss',
                                       patience=4,
                                       verbose=1,
                                       min_delta=1e-4),
                         ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           patience=2,
                                           cooldown=2,
                                           verbose=1),
                         ModelCheckpoint(filepath=kfold_weights_path,
                                         monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=True)]

            self.model.fit_generator(
                generator=train_generator(),
                steps_per_epoch=math.ceil(nTrain / float(self.batch_size)),
                epochs=self.epochs,
                verbose=1,
                callbacks=callbacks,
                validation_data=valid_generator(),
                validation_steps=math.ceil(nValid / float(self.batch_size)))

            if os.path.isfile(kfold_weights_path):
                self.model.load_weights(kfold_weights_path)

            # p_valid = model.predict_generator(
            #     val_generator, steps=len(test_index)/batch_size,
            # )
            # p_valid = model.predict(val_generator.x/255.0, batch_size = batch_size, verbose=2)
            p_valid = self.model.predict_generator(generator=valid_generator(),
                                              steps=math.ceil(nValid / float(self.batch_size)))

            ## find best threshold
            y_valid = []
            for mask_val_path in valid_y:
                # j = np.random.randint(self.nAug)
                img = cv2.imread(mask_val_path)
                img = cv2.resize(img, (self.input_dim, self.input_dim))
                # img = transformations2(img, j)
                y_valid.append(img)

            y_valid = np.array(y_valid, np.float32)

            best_score, best_threshold = find_best_seg_thr(y_valid, p_valid)
            thres.append(best_threshold)
            val_score += best_score

        return thres, val_score/float(self.nfolds)


    def test(self, thres, val_score, early_fusion=True):
        nTest = len(self.test_imgs)
        print('Testing on {} samples'.format(nTest))

        def test_generator(transformation):
            while True:
                for start in range(0, nTest, self.batch_size):
                    x_batch = []
                    end = min(start + self.batch_size, nTest)

                    for i in range(start, end):
                        img = cv2.imread(self.test_imgs[i])
                        img = cv2.resize(img, (self.input_dim, self.input_dim))
                        img = transformations2(img, transformation)
                        x_batch.append(img)
                    x_batch = np.array(x_batch, np.float32)
                    yield x_batch

        y_full_test = []
        for i in xrange(self.nfolds):
            kfold_weights_path = os.path.join('', 'weights_kfold_' + str(i + 1) + '.h5')
            if os.path.isfile(kfold_weights_path):
                self.model.load_weights(kfold_weights_path)

                # 2-fold TTA
                p_full_test = []
                for i in range(self.nTTA):
                    p_test = self.model.predict_generator(generator=test_generator(transformation=i),
                                                     steps=math.ceil(nTest / float(self.batch_size)))
                    if i % 2 == 0:
                        p_full_test.append(p_test)
                    else:
                        p_full_test.append(np.fliplr(p_test))

                p_test = np.array(p_full_test[0])
                for i in range(1, self.nTTA):
                    p_test += np.array(p_full_test[i])
                p_test /= self.nTTA
                y_full_test.append(p_test)


        def get_result(img, thresh):
            img[img > thresh] = 1
            img[img <= thresh] = 0
            return img

        raw_result = np.zeros(y_full_test[0].shape)
        if early_fusion:
            thresh = 0
            for i in xrange(self.nfolds):
                raw_result += y_full_test[i]
                thresh += thres[i]

            print raw_result

            raw_result /= float(self.nfolds)
            thresh /= float(self.nfolds)
            result = get_result(raw_result, thresh)
        else:
            for i in xrange(self.nfolds):
                raw_result += get_result(y_full_test[i], thres[i])
            result = raw_result / float(self.nfolds)



    def create_submission(best_score, avg_mask):
        print('Create submission...')
        t = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
        str = rle(avg_mask)
        t['rle_mask'] = str
        t.to_csv('subm_{}.gz'.format(best_score), index=False, compression='gzip')

if __name__ == "__main__":
    ccs = CarvanaCarSeg()

    thresh, val_score = ccs.train()
    # thresh, val_score = load_param()
    print("thresh:\n{}".format(thresh))
    print("val_score:", val_score)

    ccs.test(thres=thresh, val_score=val_score, early_fusion=True)

    # af.refine(thresh, val_score)
