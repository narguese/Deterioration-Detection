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

Epoch 1/100
  6/147 [>.............................] - ETA: 8:29 - loss: 3.6664 - accuracy: 0.1354 - precision: 0.1408 - recall: 0.0521     

---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
Input In [12], in <cell line: 1>()
----> 1 history = model.fit(train_generator,
      2                             epochs=epochs,
      3                             batch_size=batch_size,
      4                             validation_data = validation_generator)

File ~\AppData\Roaming\Python\Python39\site-packages\keras\utils\traceback_utils.py:64, in filter_traceback.<locals>.error_handler(*args, **kwargs)
     62 filtered_tb = None
     63 try:
---> 64   return fn(*args, **kwargs)
     65 except Exception as e:  # pylint: disable=broad-except
     66   filtered_tb = _process_traceback_frames(e.__traceback__)

File ~\AppData\Roaming\Python\Python39\site-packages\keras\engine\training.py:1409, in Model.fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
   1402 with tf.profiler.experimental.Trace(
   1403     'train',
   1404     epoch_num=epoch,
   1405     step_num=step,
   1406     batch_size=batch_size,
   1407     _r=1):
   1408   callbacks.on_train_batch_begin(step)
-> 1409   tmp_logs = self.train_function(iterator)
   1410   if data_handler.should_sync:
   1411     context.async_wait()

File ~\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\util\traceback_utils.py:150, in filter_traceback.<locals>.error_handler(*args, **kwargs)
    148 filtered_tb = None
    149 try:
--> 150   return fn(*args, **kwargs)
    151 except Exception as e:
    152   filtered_tb = _process_traceback_frames(e.__traceback__)

File ~\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\eager\def_function.py:915, in Function.__call__(self, *args, **kwds)
    912 compiler = "xla" if self._jit_compile else "nonXla"
    914 with OptionalXlaContext(self._jit_compile):
--> 915   result = self._call(*args, **kwds)
    917 new_tracing_count = self.experimental_get_tracing_count()
    918 without_tracing = (tracing_count == new_tracing_count)

File ~\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\eager\def_function.py:947, in Function._call(self, *args, **kwds)
    944   self._lock.release()
    945   # In this case we have created variables on the first call, so we run the
    946   # defunned version which is guaranteed to never create variables.
--> 947   return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
    948 elif self._stateful_fn is not None:
    949   # Release the lock early so that multiple threads can perform the call
    950   # in parallel.
    951   self._lock.release()

File ~\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\eager\function.py:2453, in Function.__call__(self, *args, **kwargs)
   2450 with self._lock:
   2451   (graph_function,
   2452    filtered_flat_args) = self._maybe_define_function(args, kwargs)
-> 2453 return graph_function._call_flat(
   2454     filtered_flat_args, captured_inputs=graph_function.captured_inputs)

File ~\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\eager\function.py:1860, in ConcreteFunction._call_flat(self, args, captured_inputs, cancellation_manager)
   1856 possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
   1857 if (possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE
   1858     and executing_eagerly):
   1859   # No tape is watching; skip to running the function.
-> 1860   return self._build_call_outputs(self._inference_function.call(
   1861       ctx, args, cancellation_manager=cancellation_manager))
   1862 forward_backward = self._select_forward_and_backward_functions(
   1863     args,
   1864     possible_gradient_type,
   1865     executing_eagerly)
   1866 forward_function, args_with_tangents = forward_backward.forward()

File ~\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\eager\function.py:497, in _EagerDefinedFunction.call(self, ctx, args, cancellation_manager)
    495 with _InterpolateFunctionError(self):
    496   if cancellation_manager is None:
--> 497     outputs = execute.execute(
    498         str(self.signature.name),
    499         num_outputs=self._num_outputs,
    500         inputs=args,
    501         attrs=attrs,
    502         ctx=ctx)
    503   else:
    504     outputs = execute.execute_with_cancellation(
    505         str(self.signature.name),
    506         num_outputs=self._num_outputs,
   (...)
    509         ctx=ctx,
    510         cancellation_manager=cancellation_manager)

File ~\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\eager\execute.py:54, in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     52 try:
     53   ctx.ensure_initialized()
---> 54   tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
     55                                       inputs, attrs, num_outputs)
     56 except core._NotOkStatusException as e:
     57   if name is not None:

KeyboardInterrupt: 

plot

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
