from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Input
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import  BatchNormalization
import modelnet40
from keras.models import Model
import numpy
from keras import backend as K
from keras.layers.merge import maximum

epochs = 15
batch_size = 16

# input_tensor = Input(shape=(1, 1, 2048))
input_instances = []
base_model = applications.ResNet50(weights = 'imagenet',include_top=False)
models = []
input_t = Input((224,224,3))
# print(input_t)
for i in range(0,12):
    input_instances.append(Input(shape=(224,224,3)))
    models.append(Model(inputs = base_model.input, outputs=base_model.layers[30].output)(input_instances[i]))
# print(input_instances)
# print(models)
# for tensor in models:
#     max_tensor = K.maximum(max_tensor,tensor)
max_tensor = maximum(models)
# print(max_tensor)
x = Conv2D(filters=40, kernel_size = 5)(max_tensor)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(40, activation='softmax')(x)

# model = Model(inputs= input_tensor, outputs= top_model())
model = Model(inputs= input_instances, outputs= x)
# build the ResNet50 network

# model = applications.VGG16(weights='imagenet', include_top=False)
print('Model loaded.')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
(train_generator,dataset_size_train) = modelnet40.modelnet40_generator('train',single = False, src_dir='view')
print(dataset_size_train)
(validation_generator, dataset_size_val) = modelnet40.modelnet40_generator('test', single = False, src_dir='view')

datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=90)
def batch_generator(generator, batch_size):
    # count = size
    def generator_func(generator, batch_size):
        # nonlocal count
        while True:
            i = 0
            xx=[]
            for t in range(12):
                xx.append(numpy.empty((batch_size*6,224,224,3)))
            yy = numpy.empty((batch_size*6,40))


            while i < batch_size:
                (x,y) = generator.__next__()
                # print(x)

                k = 0
                while k < 12:
                    j = 0
                    for X_batch, y_batch in datagen.flow(x[k], y, batch_size=1):
                        xx[k][i*6+j] = X_batch[0].reshape(224, 224, 3)
                        yy[i*6+j] = y[0]
                        j += 1
                    # plt.imshow(X_batch[0].reshape(224, 224, 3))
                    #     # show the plot
                    # plt.show()
                        if j == 6:
                            break
                    k+=1
                i+=1
                # count-=k
            # print(xx)
            yield xx,yy
    return (generator_func(generator, batch_size))

# def batch_generator(generator, batch_size):
#     # count = size
#     def generator_func(generator, batch_size):
#         # nonlocal count
#         while True:
#             i = batch_size
#             xx = numpy.empty((batch_size, 224,224,3))
#             yy = numpy.empty((batch_size, 40))
#             while count and i:
#                 (x,y) = generator.__next__()
#                 # print(x)
#                 xx[batch_size-i]= x[0]
#
#                 yy[batch_size-i] = y[0]
#                 i-=1
#                 count-=1
#             # print(xx)
#             yield xx,yy
#     return (generator_func(generator, batch_size))

# fine-tune the model
model.fit_generator(
    batch_generator(train_generator, batch_size),
    steps_per_epoch=dataset_size_train//batch_size,
    epochs=epochs,
    validation_data=batch_generator(validation_generator, batch_size),
    nb_val_samples=dataset_size_val//batch_size
)

model.save('my_model.h5')