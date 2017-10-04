
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input
import modelnet40
from keras.models import Model
import numpy

epochs = 15
batch_size = 16

# input_tensor = Input(shape=(224,224,3))
base_model = applications.ResNet50(weights = 'imagenet',include_top=False, input_shape=(224,224,3))
# print (base_model.input_shape)
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(40, activation='softmax'))

# model = Model(inputs= input_tensor, outputs= top_model())
# print(base_model.input)
model = Model(inputs= base_model.input, outputs= top_model(base_model.output))
# build the ResNet50 network

# model = applications.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')
# set the first 0.7 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
p = 0.7

for layer in model.layers[:int(len(model.layers)*0.7)]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

(train_generator,dataset_size_train) = modelnet40.modelnet40_generator('train',src_dir='view')
(validation_generator, dataset_size_val) = modelnet40.modelnet40_generator('test', src_dir='view')

#Data augmentation
datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=90)
def batch_generator(generator, batch_size):
    # count = size
    def generator_func(generator, batch_size):
        # nonlocal count
        while True:
            i = 0
            xx = numpy.empty((batch_size*6, 224,224,3))
            yy = numpy.empty((batch_size*6, 40))
            while i < batch_size:
                (x,y) = generator.__next__()
                # print(x)
                j = 0
                for X_batch, y_batch in datagen.flow(x, y, batch_size=1):
                    xx[i*6+j] = X_batch[0].reshape(224, 224, 3)
                    yy[i*6+j] = y[0]
                    j += 1
                    if j == 6:
                        break
                i+=1
                # count-=1
            # print(xx)
            yield xx,yy
    return (generator_func(generator, batch_size))
#Without augmentation
# def batch_generator(generator, batch_size):
#     # count = size
#     def generator_func(generator, batch_size):
#         # nonlocal count
#         while True:
#
#             i = batch_size
#             xx = numpy.empty((batch_size, 224,224,3))
#             yy = numpy.empty((batch_size, 40))
#             while i:
#
#                 (x,y) = generator.__next__()
#                 # print(x)
#                 xx[batch_size-i]= x[0]
#
#                 yy[batch_size-i] = y[0]
#                 i-=1
#                 # count-=1
#             # print(xx)
#             yield xx,yy
#     return (generator_func(generator, batch_size))

# fine-tune the model
model.fit_generator(
    batch_generator(train_generator, batch_size),
    steps_per_epoch=dataset_size_train //batch_size,
    epochs=epochs,
    validation_data=batch_generator(validation_generator, batch_size),
    nb_val_samples=dataset_size_val //batch_size
)
model.save('my_model.h5')