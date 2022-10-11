import cv2 as cv
import numpy as np
from sklearn.utils import shuffle
import os


def prepare_images(path_images_c, path_images_s, path_save, imsize):
    print('Reading and preprocessing images.')

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    x = read_x(path_images_c, imsize, num_samples=20)
    for i in range(len(x)):
        cv.imwrite('{}/fashion_{}.jpeg'.format(path_save, i), x[i])

    x = read_x(path_images_s, imsize)
    for i in range(len(x)):
        cv.imwrite('{}/style_{}.jpeg'.format(path_save, i), x[i])


def read_x(path_images, imsize, num_samples=None):
    array_of_images = []
    paths = shuffle(os.listdir(path_images), random_state=5)
    sample_i = 0
    for _, file in enumerate(paths):
        fname = path_images + '/' + file
        if fname.endswith('.jpg') or fname.endswith('.jpeg'):
            image = cv.imread(fname)
            single_array = cv.resize(image, imsize, interpolation=cv.INTER_AREA)
            array_of_images.append(single_array)
            sample_i += 1
            if num_samples is not None and sample_i == num_samples:
                break
    return np.array(array_of_images)



def prepare_segmentation(path_images, path_save):
    print('Preparing segmentation images.')

    if not os.path.exists(path_save):
        os.mkdir(path_save)

    for _, file in enumerate(os.listdir(path_images)):
        fname = path_images + '/' + file
        if fname.endswith('.jpeg'):
            img = cv.imread(fname)
            seg = np.zeros(img.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (1, 1, img.shape[0], img.shape[1])

            cv.grabCut(img, seg, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

            seg2 = np.where((seg == 2) | (seg == 0), 0, 255).astype('uint8')
            seg_rgb = cv.cvtColor(seg2, cv.COLOR_GRAY2RGB)
            cv.imwrite('{}/segmentation_{}'.format(path_save, file), seg_rgb)