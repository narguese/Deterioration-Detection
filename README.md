# Deterioration-Detection
This dataset includes two materials, stone and brick, with damages from historical buildings. Various deep learning models (e.g., Mobilenetv2, Inception Resnet, and Resnet50) were used for analysis, with InceptionResnet achieving a 96% accuracy score
import libreries

import matplotlib.pyplot as plt

import numpy as np

import os

import PIL

import tensorflow as tf

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image_dataset_from_directory

​

​

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from tensorflow.keras import layers

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import plot_model

​

from zipfile import ZipFile

directories

train_dir = 'D:/historical brigde/MAIN DATA FOR CLASSIFICATION/TEST TEST AUGMENT STONE/train'

test_dir = 'D:/historical brigde/MAIN DATA FOR CLASSIFICATION/TEST TEST AUGMENT STONE/test'

image information

img_width, img_height = 299, 299

epochs = 200

batch_size = 32

Define Channel of images

if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)

Read traindata from directory

train_dataset = image_dataset_from_directory (train_dir,

                                                            shuffle=True,

                                                            validation_split=0.2,

                                                            subset="training",

                                                            seed=42,

                                                            batch_size=batch_size,

                                                            image_size=(img_width, img_height))

Found 5842 files belonging to 7 classes.
Using 4674 files for training.

Read validdata from directory of traindata

val_dataset = image_dataset_from_directory (train_dir,

                                            shuffle=True,

                                            validation_split=0.2,

                                            subset="validation",

                                            seed=42,

                                            batch_size=batch_size,

                                            image_size=(img_width, img_height))

Found 5842 files belonging to 7 classes.
Using 1168 files for validation.

Read testdata from directory

test_dataset = image_dataset_from_directory (test_dir,

                                            shuffle=True,

                                            batch_size= batch_size,

                                            image_size=(img_width, img_height))

Found 2489 files belonging to 7 classes.

Display the dataset

class_names = train_dataset.class_names

print (class_names)

['B-Cracking', 'B-Erosion', 'B-No defect', 'B-Salt Efflorescence', 'S-Cracking', 'S-Flaking', 'S-No defect']

plt.figure(figsize=(10, 10))

for images, labels in train_dataset.take(1):

  for i in range(9):

    ax = plt.subplot(3, 3, i + 1)

    plt.imshow(images[i].numpy().astype("uint8"))

    plt.title(class_names[labels[i]])

    plt.axis("off")

ImageDataGenerator

#datatrain throgh ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=50,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   zoom_range=0.3,

                                   horizontal_flip=True,

                                   vertical_flip=True,

                                   fill_mode='constant',

                                   cval=0,

                                   rescale=1./255,

                                   validation_split=0.2)

#datatest generat

test_datagen = ImageDataGenerator(rescale=1. / 255)

​

#reading traindata from directory

train_generator = train_datagen.flow_from_directory(train_dir,

                                                    shuffle=True,

                                                    seed=42,

                                                    target_size=(img_width, img_height),

                                                    batch_size=batch_size,

                                                    class_mode='categorical',

                                                    subset='training'

                                                   )

​

#reading validdata from directory of traindata

validation_generator = train_datagen.flow_from_directory(train_dir,

                                                        shuffle=True,

                                                        seed=42,

                                                        target_size=(img_width, img_height),

                                                        batch_size=batch_size,

                                                        class_mode='categorical',

                                                        subset='validation'

                                                       )

​

#reading testdata from directory

test_generator = test_datagen.flow_from_directory(test_dir,

                                                  shuffle=True,

                                                  target_size=(img_width, img_height),

                                                  batch_size=batch_size,

                                                  class_mode='categorical'

                                                 )

Found 4676 images belonging to 7 classes.
Found 1166 images belonging to 7 classes.
Found 2489 images belonging to 7 classes.

inception_resnet_v2

model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(

    input_shape=input_shape,

    include_top=False,

    weights='imagenet',

    classifier_activation='softmax')

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 399, 399, 32)      416       
                                                                 
 activation (Activation)     (None, 399, 399, 32)      0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 199, 199, 32)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 198, 198, 32)      4128      
                                                                 
 activation_1 (Activation)   (None, 198, 198, 32)      0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 99, 99, 32)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 98, 98, 64)        8256      
                                                                 
 activation_2 (Activation)   (None, 98, 98, 64)        0         
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 49, 49, 64)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 153664)            0         
                                                                 
 dense (Dense)               (None, 64)                9834560   
                                                                 
 activation_3 (Activation)   (None, 64)                0         
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 7)                 455       
                                                                 
 activation_4 (Activation)   (None, 7)                 0         
                                                                 
=================================================================
Total params: 9,847,815
Trainable params: 9,847,815
Non-trainable params: 0
_________________________________________________________________
You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model/model_to_dot to work.

# The last 15 layers fine tune

for layer in model.layers[:-15]:

    layer.trainable = False

x = model.output

x = GlobalAveragePooling2D()(x)

x = Flatten()(x)

x = Dense(units=128, activation='relu')(x)

x = Dropout(0.3)(x)

x = Dense(units=128, activation='relu')(x)

x = Dropout(0.3)(x)

output  = Dense(units=7, activation='softmax')(x)

model = Model(model.input, output)

#summary model

model.summary()

#plot model

tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

Compile Model

model.compile(loss='categorical_crossentropy',

              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

train model

history = model.fit(train_generator,

                            epochs=epochs,

                            batch_size=batch_size,

                            validation_data = validation_generator)


# Displaying curves of loss and accuracy during training

import matplotlib.pyplot as plt

​

def plot_history(history):

  accuracy = history.history['accuracy']

  val_accuracy = history.history['val_accuracy']

  loss = history.history['loss']

  val_loss = history.history['val_loss']

​

  epochs = range(1, len(accuracy) + 1)

​

  plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

  plt.plot(epochs, val_accuracy, 'r', label='Val accuracy')

  plt.title('Training and validation accuracy')

  plt.legend()

​

  plt.figure()

​

  plt.plot(epochs, loss, 'bo', label='Training loss')

  plt.plot(epochs, val_loss, 'r', label='Val loss')

  plt.title('Training and val loss')

  plt.legend()

​

  plt.show()

​

plot_history(history)

model.evaluate(test_generator, verbose=1)

save weights

model.save_weights('InceptionResNetV2_model_weight.h5')

model.save('InceptionResNetV2_model_keras.h5')

predict an image

from keras.models import save_model,load_model

from PIL import Image

import cv2

import numpy as np

from google.colab.patches import cv2_imshow

​

# model = load_model('model_weight.h5')

​

# model.compile(loss='categorical_crossentropy',

#               optimizer='Adam',

#               metrics=['accuracy'])

​

img = cv2.imread('predict.jpg')

print('Original Dimensions : ',img.shape)

​

width = 299

height = 299

dim = (width, height)

​

resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

print('resized Dimensions : ',resized_img.shape)

​

cv2_imshow(resized_img)

cv2.waitKey(0)

​

img_reshape = np.reshape(resized_img,[1,299,299,3])

​

from keras.preprocessing import image

​

img_rescale = np.array(img_reshape)

img_rescale = img_rescale.astype('float32')

img_rescale /=255

​

#predict model

saved_model = load_model('Mobilenet_model_keras.h5', compile=False)

saved_model.compile(optimizer='adam',

          loss='categorical_crossentropy',

          metrics=['accuracy'])

predict = saved_model.predict([img_rescale])

​

print([predict])

print(class_names)
