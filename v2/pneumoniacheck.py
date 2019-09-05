from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np

# Evaluations
from sklearn.metrics import classification_report, confusion_matrix

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


print('Teach CNN? [y/n]')
# input
todo1 = input()

# todo1 = 'y'

if (todo1 == 'y'):

    # Build the CNN
    classifier = Sequential()

    # Convolution
    classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

    # Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Pooling is made with a 2x2 array
    # Add 2nd convolutional layer with the same structure as the 1st to improve predictions
    classifier.add(Conv2D(32, (3, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Flattening
    classifier.add(Flatten())

    # Full Connection
    classifier.add(Dense(activation = 'relu', units = 128))
    classifier.add(Dense(activation = 'sigmoid', units = 1))

    # Compile the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Image Augmentation
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    # Apply several transformations to train the model in a better significant way, keras documentation provides
    # all the required information for augmentation
    test_datagen = ImageDataGenerator(rescale = 1./255)

    print('Adding data sets')

    training_set = train_datagen.flow_from_directory('./chest_xray/train',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    classifier.summary()

    print('Running fit_generator')
    print('You can go for coffee. Or for lunch. Or to smoke. Or all of this, you will have a lot of time now...')

    history = classifier.fit_generator(training_set,
                            steps_per_epoch = 163,
                            epochs = 20,
                            validation_data = test_set,
                            validation_steps = 624)


    print('Save the model? [y/n]')

    # input
    todo2 = input()

    # todo2 = 'y'

    if (todo2 == 'y'):

        # serialize classifier to JSON
        model_json = classifier.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        classifier.save_weights("model.h5")
        print("Model saved")

    #Accuracy
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Test set'], loc='upper left')
    plt.savefig("Accuracy.png")
    plt.clf()

    #Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Test set'], loc='upper left')
    plt.savefig("Loss.png")
    plt.clf()

print('Load the model? [y/n]')

# input

todo3 = input()

# todo3 = 'y'

if (todo3 == 'y'):

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights("model.h5")
    print("Loaded model from disk")

if ((todo1 != 'y') and (todo3 != 'y')):

    print('Then why are you here? You need help, dude...')
    exit()


# print('Test the model? [y/n]')
# # input
# todo4 = input()

# todo4 = 'y'

if (todo4 == 'y'):
    # desired_batch_size = 8

    # val_datagen = ImageDataGenerator(rescale = 1./255)
    # val_set = val_datagen.flow_from_directory('./chest_xray/val',
    #                                             target_size = (64, 64),
    #                                             batch_size = desired_batch_size,
    #                                             class_mode = 'binary')
    # filenames = val_set.filenames
    # nb_samples = len(filenames)

    # print(nb_samples)

    # # probabilities = classifier.predict_generator(val_set, steps=nb_samples)
    # probabilities = classifier.predict_generator(val_set, steps = np.ceil(nb_samples/desired_batch_size))
    

    # print(probabilities)
    # print(val_set.classes)

    # y_pred = np.argmax(probabilities, axis=1)

    # print('Confusion Matrix')
    # print(confusion_matrix(val_set.classes, y_pred))
    # print('Classification Report')
    # target_names = ['Normal', 'Pneumonia']
    # print(classification_report(val_set.classes, y_pred, target_names=target_names))
    
    # predicting images

    # print(classifier)

    # img = image.load_img('./chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg', target_size=(64, 64))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    
    # preds = classifier.predict(x)
    # print("NORMAL =", preds)

    # img = image.load_img('./chest_xray/val/PNEUMONIA/person1952_bacteria_4883.jpeg', target_size=(64, 64))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    
    # preds = classifier.predict(x)
    # print("PNEUMONIA =", preds)

    # img = image.load_img('./chest_xray/val/NORMAL/NORMAL2-IM-1431-0001.jpeg', target_size=(64, 64))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    
    # preds = classifier.predict(x)
    # print("NORMAL =", preds)

    # img = image.load_img('./chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg', target_size=(64, 64))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    
    # preds = classifier.predict(x)
    # print("PNEUMONIA =", preds)

    # img = image.load_img('./chest_xray/val/NORMAL/NORMAL2-IM-1430-0001.jpeg', target_size=(64, 64))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    
    # preds = classifier.predict(x)
    # print("NORMAL =", preds)





# Using TensorFlow backend.
# WARNING:tensorflow:From C:\Users\ftokarev\AppData\Local\Continuum\anaconda3\envs\NN_Env\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Colocations handled automatically by placer.
# Found 5216 images belonging to 2 classes.
# Found 624 images belonging to 2 classes.
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 62, 62, 32)        896
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 31, 31, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 29, 29, 32)        9248
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 6272)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 128)               802944
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 129
# =================================================================
# Total params: 813,217
# Trainable params: 813,217
# Non-trainable params: 0

# Epoch 1/20
# 2019-09-05 02:15:13.993288: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
# To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
# 2019-09-05 02:15:13.993787: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
# 163/163 [==============================] - 316s 2s/step - loss: 0.4211 - acc: 0.8117 - val_loss: 0.4413 - val_acc: 0.7784
# Epoch 2/20
# 163/163 [==============================] - 250s 2s/step - loss: 0.2291 - acc: 0.9045 - val_loss: 0.3890 - val_acc: 0.8198
# Epoch 3/20
# 163/163 [==============================] - 248s 2s/step - loss: 0.2162 - acc: 0.9076 - val_loss: 0.3338 - val_acc: 0.8684
# Epoch 4/20
# 163/163 [==============================] - 250s 2s/step - loss: 0.1882 - acc: 0.9208 - val_loss: 0.3086 - val_acc: 0.8732
# Epoch 5/20
# 163/163 [==============================] - 244s 1s/step - loss: 0.1827 - acc: 0.9294 - val_loss: 0.3645 - val_acc: 0.8495
# Epoch 6/20
# 163/163 [==============================] - 245s 2s/step - loss: 0.1653 - acc: 0.9350 - val_loss: 0.4391 - val_acc: 0.8300
# Epoch 7/20
# 163/163 [==============================] - 246s 2s/step - loss: 0.1568 - acc: 0.9402 - val_loss: 0.3363 - val_acc: 0.8814
# Epoch 8/20
# 163/163 [==============================] - 241s 1s/step - loss: 0.1634 - acc: 0.9375 - val_loss: 0.4293 - val_acc: 0.8449
# Epoch 9/20
# 163/163 [==============================] - 240s 1s/step - loss: 0.1500 - acc: 0.9417 - val_loss: 0.2948 - val_acc: 0.8927
# Epoch 10/20
# 163/163 [==============================] - 239s 1s/step - loss: 0.1615 - acc: 0.9379 - val_loss: 0.4510 - val_acc: 0.8204
# Epoch 11/20
# 163/163 [==============================] - 243s 1s/step - loss: 0.1488 - acc: 0.9433 - val_loss: 0.5504 - val_acc: 0.8030
# Epoch 12/20
# 163/163 [==============================] - 244s 1s/step - loss: 0.1465 - acc: 0.9415 - val_loss: 0.3640 - val_acc: 0.8751
# Epoch 13/20
# 163/163 [==============================] - 243s 1s/step - loss: 0.1406 - acc: 0.9471 - val_loss: 0.2853 - val_acc: 0.8959
# Epoch 14/20
# 163/163 [==============================] - 243s 1s/step - loss: 0.1413 - acc: 0.9433 - val_loss: 0.4722 - val_acc: 0.8397
# Epoch 15/20
# 163/163 [==============================] - 244s 1s/step - loss: 0.1278 - acc: 0.9519 - val_loss: 0.2726 - val_acc: 0.9021
# Epoch 16/20
# 163/163 [==============================] - 244s 1s/step - loss: 0.1213 - acc: 0.9525 - val_loss: 0.3967 - val_acc: 0.8687
# Epoch 17/20
# 163/163 [==============================] - 244s 1s/step - loss: 0.1191 - acc: 0.9563 - val_loss: 0.4358 - val_acc: 0.8397
# Epoch 18/20
# 163/163 [==============================] - 245s 2s/step - loss: 0.1225 - acc: 0.9526 - val_loss: 0.3605 - val_acc: 0.8655
# Epoch 19/20
# 163/163 [==============================] - 245s 2s/step - loss: 0.1217 - acc: 0.9544 - val_loss: 0.4480 - val_acc: 0.8648
# Epoch 20/20
# 163/163 [==============================] - 243s 1s/step - loss: 0.1223 - acc: 0.9523 - val_loss: 0.2973 - val_acc: 0.9057
