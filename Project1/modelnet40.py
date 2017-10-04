
"""
Multiview (MVCNN) ModelNet-40 dataset for Keras
"""

import os, os.path
import pprint
import glob
import random
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
import numpy
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

DEFAULT_SRCDIR = 'modelnet40'
DEFAULT_TARGET_SIZE = (224, 224)        # Input size for ResNet-50
nclasses = 40
nviews = 12

def subdirs(dirname):
    return [x for x in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, x))]

def read_image(filename, target_size, preprocess=None):
#    print('read_image:', filename)
    x = image.load_img(filename, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    if preprocess is not None:
        x = preprocess(x)
    #print('read_image, shape:', x.shape)
    return x

def modelnet40_filenames(subset, src_dir=DEFAULT_SRCDIR):
    """
    List of models for ModelNet-40.
    
    Each model is a pair (class_index, filename_L).
    Here filename_L is a 12 length list of image filenames of views of the model.
    """
    src_dir = os.path.join(src_dir, 'classes')
    classes = sorted(subdirs(src_dir))
    ans = []
    for (icls, cls) in enumerate(classes):
        subset_dir = os.path.join(src_dir, cls, subset)
        model_dirs = subdirs(subset_dir)
        for model_dir in model_dirs:
            filenames = glob.glob(os.path.join(src_dir, cls, subset, model_dir, '*.png'))
            ans.append((icls, filenames))
            # print(ans)
    return ans

def modelnet40_generator(subset, src_dir=DEFAULT_SRCDIR, single=True, target_size=DEFAULT_TARGET_SIZE, repeats=None, shuffle=True, verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input):
    """
    A generator that returns images and classes from ModelNet-40 in size one batches.
    
    Returns (g, dataset_size), where g is the generator and dataset_size is the number of elements in the dataset.
    
    The generator yields by default an infinite number of elements which each have the form (x, y), where x is
    an input for supervised training and y is an output. If single is True (single view mode) then x is a 4D numpy
    array with shape 1 x h x w x 3, representing an input image, and y is a 2D tensor of shape 1 x nclasses, where
    nclasses is the number of classes in ModelNet-40 (defined as the global nclasses, equal to 40). If single is False
    (multiple view mode) then x is a list of arrays representing different views of the same model: the list has length
    nviews (defined as the global nviews, equal to 12): each view is an numpy array of an image with shape 1 x h x w x 3.
    
    Arguments:
    - subset:       Either 'train' or 'test'
    - src_dir:      The ModelNet-40 directory.
    - single:       If true, return one image and class at a time in the format (img, cls).
                    If false, return a list of (img, cls) for all (12) views.
    - repeats:      Number of times to repeat the dataset. If None, repeat forever.
    - shuffle:      If true, randomly shuffle the dataset.
    - verbose:      Print information about loading: verbose=0: no info, 1 is some info, verbose=2 is more info.
    - frac:         Fraction of dataset to load (use frac < 1.0 for quick tests).
    - class_array:  If true, return class as a 1-hot vector array.
    - preprocess:   Preprocessing function from a Keras model to be called on each image (numpy array) after being read.
                    (or None, the default, for no preprocessing).
    """
    viewL = modelnet40_filenames(subset, src_dir)

    def generator_func():
        repeat = 0
        while repeats is None or repeat < repeats:
            if shuffle:
                random.shuffle(viewL)
            for (i, view) in enumerate(viewL[:int(len(viewL)*frac)]):
                if verbose == 1 and i % 100 == 0:
                    print('Loading %s data: %.1f%%' % (subset, i*100.0/(len(viewL)*frac)))
                (cls, view) = view
                if verbose == 2:
                    print('Loading data point %d, cls = %d' % (i, cls))
                if class_array:
                    cls_array = numpy.zeros((1, nclasses), 'float32')
                    cls_array[0, cls] = 1.0
                    cls = cls_array
                if single:
                    filename = random.choice(view)
                    yield (read_image(filename, target_size, preprocess), cls)
                else:
                    yield ([read_image(view_elem, target_size, preprocess) for view_elem in view], cls)
            repeat += 1 
    return (generator_func(), len(viewL))
def batch_generator(generator, batch_size, size):
    # count = size
    def generator_func(generator, batch_size):
        # nonlocal count
        while True:

            i = batch_size
            xx = numpy.empty((batch_size, 224,224,3))
            yy = numpy.empty((batch_size, 40))
            while i:

                (x,y) = generator.__next__()
                # print(x)
                xx[batch_size-i]= x[0]

                yy[batch_size-i] = y[0]
                i-=1
                # count-=1
            # print(xx)
            yield xx,yy
    return (generator_func(generator, batch_size))
if __name__ == '__main__':

    (g, dataset_size) = modelnet40_generator('test', single=True, frac = 0.01, src_dir='view')
    print(dataset_size)

    g2 = batch_generator(g,500,dataset_size)
    # print('Loading first element from dataset')
    i = 0
    while True:
        (x1, y1) = g2.__next__()
        i+=1# Python 3 syntax. Use .next() instead for Python 2.
        print(i)
        # print(len(x1))
        # print(x1[499])
        # print(y1)
    # plt.imshow(x1[0], cmap=plt.get_cmap('gray'))
    # plt.imshow(x1[0])
    # plt.show()
    # (x2,y2) = g.__next__()
    # plt.imshow(x2[0])
    # plt.show()
    # im = Image.open('lena.jpg')
    # im = Image.fromarray(x1[0])
    # im.show()
    # a = numpy.array(im)
    # print(a)

    # datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=90)
    # # fit parameters from data
    # datagen.fit(x1)
    # # Configure batch size and retrieve one batch of images
    # i = 0
    # for X_batch, y_batch in datagen.flow(x1, y1, batch_size=1):
    #     # Show 9 images
    #     for i in range(0, 6):
    #         print(i)
            # pyplot.subplot(330 + 1 + i)
        # i+=1
        # plt.imshow(X_batch[0].reshape(224, 224, 3))
    #     # show the plot
    #     plt.show()
    #     if i == 6:
            # break
    # print(x1.shape)
    # print('Loading second element from dataset')
    # (x2, y2) = g.__next__()                             # Python 3 syntax. Use .next() instead for Python 2.
    # print(x2, y2)
    print('Done loading')
