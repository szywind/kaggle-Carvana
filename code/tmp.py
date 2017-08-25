import numpy as np
import torch
import cv2
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd.variable import Variable

INPUT_PATH = 'input/'
# img = cv2.imread('./input/train/0cdf5b5d0ce1_01.jpg')
img = np.array(Image.open(INPUT_PATH + 'train_masks/{}_mask.gif'.format('0cdf5b5d0ce1_01')), dtype=np.uint8)

# img = (img/255.0).transpose(2,0,1)[np.newaxis,...] # 1x3xhxw
img = img[np.newaxis, np.newaxis, ...].astype(np.float32)
x = torch.from_numpy(img)
x = F.avg_pool2d(x, kernel_size=11, padding=5, stride=1)
ind =x.ge(0.01) * x.le(0.99)
ind = ind.float()
weights = Variable(torch.tensor.torch.ones(x.size()))#.cuda()
# weights = torch.tensor.torch.ones(x.size())

w0 = weights.sum()
weights = weights + ind*2
w1 = weights.sum()
weights = weights/w1*w0




img = np.array(Image.open(INPUT_PATH + 'train_masks/{}_mask.gif'.format('0cdf5b5d0ce1_01')), dtype=np.uint8)
y_true = img.astype(np.float32)
a = cv2.blur(y_true, (11, 11))

# y_true = tf.stack(img[np.newaxis,...,np.newaxis].astype(np.float32))
# a = AveragePooling2D((11, 11), padding='same', strides=1)(y_true)