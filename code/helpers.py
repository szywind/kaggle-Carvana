from sklearn.metrics import fbeta_score
import numpy as np
import cv2
import imutils

def comp_mean(imglist):
    mean = [0, 0, 0]
    for img in imglist:
        mean += np.mean(np.mean(img, axis=0), axis=0)
    return mean/len(imglist)

def load_param():

    thresh = [[0.03, 0.03, 0.05, 0.07, 0.03, 0.02, 0.05, 0.03, 0.05, 0.05, 0.04, 0.03, 0.05, 0.1, 0.04, 0.04, 0.06],
     [0.05, 0.03, 0.09, 0.08, 0.03, 0.02, 0.05, 0.08, 0.04, 0.05, 0.02, 0.03, 0.03, 0.07, 0.04, 0.06, 0.05],
     [0.04, 0.03, 0.05, 0.06, 0.02, 0.04, 0.05, 0.05, 0.03, 0.05, 0.04, 0.03, 0.03, 0.11, 0.04, 0.05, 0.06],
     [0.02, 0.03, 0.06, 0.1, 0.03, 0.01, 0.06, 0.05, 0.04, 0.1, 0.05, 0.03, 0.03, 0.1, 0.04, 0.05, 0.1],
     [0.04, 0.03, 0.04, 0.06, 0.03, 0.03, 0.05, 0.09, 0.03, 0.07, 0.07, 0.04, 0.04, 0.08, 0.03, 0.06, 0.09]]
    val_score = 0.93065441478548683

    return thresh, val_score

def find_f_measure_threshold2(probs, labels, num_iters=100, seed=0.21):
    _, num_classes = labels.shape[0:2]
    best_thresholds = [seed] * num_classes
    best_scores = [0] * num_classes
    for t in range(num_classes):

        thresholds = list(best_thresholds)  # [seed]*num_classes
        for i in range(num_iters):
            th = i / float(num_iters)
            thresholds[t] = th
            f2 = fbeta_score(labels, probs > thresholds, beta=2, average='samples')
            if f2 > best_scores[t]:
                best_scores[t] = f2
                best_thresholds[t] = th
        print('\t(t, best_thresholds[t], best_scores[t])=%2d, %0.3f, %f' % (t, best_thresholds[t], best_scores[t]))
    print('')
    return best_thresholds, best_scores


def normallize(img):
    img = img.astype(np.float16)

    img[:, :, 0] = (img[:, :, 0] - 103.94) * 0.017
    img[:, :, 1] = (img[:, :, 1] - 116.78) * 0.017
    img[:, :, 2] = (img[:, :, 2] - 123.68) * 0.017
    img = np.expand_dims(img, axis=0)
    return img


def transformations(src, choice):
    if choice == 0:
        # Rotate 90
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    if choice == 1:
        # Rotate 90 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        src = cv2.flip(src, flipCode=1)
    if choice == 2:
        # Rotate 180
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
    if choice == 3:
        # Rotate 180 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
        src = cv2.flip(src, flipCode=1)
    if choice == 4:
        # Rotate 90 counter-clockwise
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    if choice == 5:
        # Rotate 90 counter-clockwise and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        src = cv2.flip(src, flipCode=1)
    return src

def transformations2(src, choice):
    mode = choice // 2
    src = imutils.rotate(src, mode * 90)
    if choice % 2 == 1:
        src = cv2.flip(src, flipCode=1)
    return src

def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    bytes = np.where(img.flatten() == 1)[0]
    runs = []
    prev = -2
    for b in bytes:
        if (b > prev + 1): runs.extend((b + 1, 0))
        runs[-1] += 1
        prev = b

    return ' '.join([str(i) for i in runs])


def dice(im1, im2, empty_score=1.0):
    im1 = im1.astype(np.bool)
    im2 = im2.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum

def get_score(train_masks, avg_masks, thr):
    d = 0.0
    for i in range(train_masks.shape[0]):
        avg_masks[i][avg_masks[i] > thr] = 1
        avg_masks[i][avg_masks[i] <= thr] = 0
        d += dice(train_masks[i], avg_masks[i])
    return d/train_masks.shape[0]

def find_best_seg_thr(masks_gt, masks_pred):
    best_score = 0
    best_thr = -1
    for t in range(400, 600):
        thr = t/1000
        score = get_score(masks_gt, masks_pred, thr)
        print('THR: {:.3f} SCORE: {:.6f}'.format(thr, score))
        if score > best_score:
            best_score = score
            best_thr = thr

    print('Best score: {} Best thr: {}'.format(best_score, best_thr))
    return best_score, best_thr