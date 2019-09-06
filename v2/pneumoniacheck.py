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
                            epochs = 32,
                            validation_data = test_set,
                            validation_steps = 624)


    print('Save the model? [y/n]')

    # input
    # todo2 = input()

    todo2 = 'n'

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

# # input

# todo3 = input()

todo3 = 'y'

if (todo3 == 'y'):

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights("model.h5")
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# if ((todo1 != 'y') and (todo3 != 'y')):

#     print('Then why are you here? You need help, dude...')
#     exit()


print('Test the model? [y/n]')
# input
# todo4 = input()

todo4 = 'y'

if (todo4 == 'y'):
    desired_batch_size = 8

    val_datagen = ImageDataGenerator(rescale = 1./255)
    val_set = val_datagen.flow_from_directory('./chest_xray/val',
                                                target_size = (64, 64),
                                                batch_size = desired_batch_size,
                                                class_mode = 'binary')
    filenames = val_set.filenames
    nb_samples = len(filenames)

    print(nb_samples)

    # probabilities = classifier.predict_generator(val_set, steps=nb_samples)
    probabilities = classifier.predict_generator(val_set, steps = np.ceil(nb_samples/desired_batch_size))
    
    y_pred = np.around(probabilities)

    print(probabilities)
    print(y_pred)
    print(val_set.classes)

    #y_pred = np.argmax(probabilities, axis=1)
    #print(y_pred)

    print('Confusion Matrix')
    print(confusion_matrix(val_set.classes, y_pred))
    print('Classification Report')
    target_names = ['Normal', 'Pneumonia']
    print(classification_report(val_set.classes, y_pred, target_names=target_names))
    
    # predicting images

    print(classifier)

    img = image.load_img('./chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("NORMAL =", preds)

    img = image.load_img('./chest_xray/val/PNEUMONIA/person1952_bacteria_4883.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("PNEUMONIA =", preds)

    img = image.load_img('./chest_xray/val/NORMAL/NORMAL2-IM-1431-0001.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("NORMAL =", preds)

    img = image.load_img('./chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("PNEUMONIA =", preds)

    img = image.load_img('./chest_xray/val/NORMAL/NORMAL2-IM-1430-0001.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("NORMAL =", preds)





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

# Epoch 1/40
# 2019-09-05 14:30:41.848441: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
# To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
# 2019-09-05 14:30:41.850868: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
# 163/163 [==============================] - 314s 2s/step - loss: 0.3452 - acc: 0.8472 - val_loss: 0.4983 - val_acc: 0.7723
# Epoch 2/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.2499 - acc: 0.8988 - val_loss: 0.4193 - val_acc: 0.8321
# Epoch 3/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.2195 - acc: 0.9124 - val_loss: 0.3073 - val_acc: 0.8728
# Epoch 4/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.1817 - acc: 0.9262 - val_loss: 0.3759 - val_acc: 0.8532
# Epoch 5/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.1670 - acc: 0.9354 - val_loss: 0.3110 - val_acc: 0.8892
# Epoch 6/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.1551 - acc: 0.9415 - val_loss: 0.2648 - val_acc: 0.8942
# Epoch 7/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.1698 - acc: 0.9340 - val_loss: 0.3339 - val_acc: 0.8638
# Epoch 8/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.1456 - acc: 0.9442 - val_loss: 0.2805 - val_acc: 0.8817
# Epoch 9/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.1354 - acc: 0.9513 - val_loss: 0.2873 - val_acc: 0.8972
# Epoch 10/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.1329 - acc: 0.9479 - val_loss: 0.3240 - val_acc: 0.8765
# Epoch 11/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.1299 - acc: 0.9494 - val_loss: 0.3013 - val_acc: 0.8959
# Epoch 12/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.1291 - acc: 0.9498 - val_loss: 0.3146 - val_acc: 0.8703
# Epoch 13/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.1254 - acc: 0.9509 - val_loss: 0.3129 - val_acc: 0.8941
# Epoch 14/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.1244 - acc: 0.9532 - val_loss: 0.2740 - val_acc: 0.9087
# Epoch 15/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.1256 - acc: 0.9505 - val_loss: 0.2978 - val_acc: 0.9055
# Epoch 16/40
# 163/163 [==============================] - 243s 1s/step - loss: 0.1168 - acc: 0.9534 - val_loss: 0.2972 - val_acc: 0.9119
# Epoch 17/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.1277 - acc: 0.9528 - val_loss: 0.4050 - val_acc: 0.8799
# Epoch 18/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.1135 - acc: 0.9544 - val_loss: 0.2761 - val_acc: 0.9131
# Epoch 19/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.1103 - acc: 0.9590 - val_loss: 0.3653 - val_acc: 0.8896
# Epoch 20/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.1073 - acc: 0.9586 - val_loss: 0.3977 - val_acc: 0.8814
# Epoch 21/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.1010 - acc: 0.9613 - val_loss: 0.3914 - val_acc: 0.8796
# Epoch 22/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.1188 - acc: 0.9544 - val_loss: 0.4199 - val_acc: 0.8660
# Epoch 23/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.0946 - acc: 0.9632 - val_loss: 0.3769 - val_acc: 0.9069
# Epoch 24/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.0974 - acc: 0.9626 - val_loss: 0.3776 - val_acc: 0.8719
# Epoch 25/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.0934 - acc: 0.9636 - val_loss: 0.2226 - val_acc: 0.9216
# Epoch 26/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.0855 - acc: 0.9653 - val_loss: 0.3248 - val_acc: 0.9050
# Epoch 27/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.0928 - acc: 0.9640 - val_loss: 0.2603 - val_acc: 0.9300
# Epoch 28/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.0835 - acc: 0.9686 - val_loss: 0.3474 - val_acc: 0.9053
# Epoch 29/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.0907 - acc: 0.9630 - val_loss: 0.3066 - val_acc: 0.8957
# Epoch 30/40
# 163/163 [==============================] - 247s 2s/step - loss: 0.0903 - acc: 0.9689 - val_loss: 0.3617 - val_acc: 0.8929
# Epoch 31/40
# 163/163 [==============================] - 243s 1s/step - loss: 0.0857 - acc: 0.9686 - val_loss: 0.3507 - val_acc: 0.9038
# Epoch 32/40
# 163/163 [==============================] - 243s 1s/step - loss: 0.0766 - acc: 0.9720 - val_loss: 0.2888 - val_acc: 0.9056
# Epoch 33/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.0906 - acc: 0.9659 - val_loss: 0.2436 - val_acc: 0.9183
# Epoch 34/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.0814 - acc: 0.9691 - val_loss: 0.2778 - val_acc: 0.9153
# Epoch 35/40
# 163/163 [==============================] - 243s 1s/step - loss: 0.0927 - acc: 0.9655 - val_loss: 0.3873 - val_acc: 0.8943
# Epoch 36/40
# 163/163 [==============================] - 244s 1s/step - loss: 0.0829 - acc: 0.9701 - val_loss: 0.3227 - val_acc: 0.9183
# Epoch 37/40
# 163/163 [==============================] - 245s 2s/step - loss: 0.0845 - acc: 0.9691 - val_loss: 0.2929 - val_acc: 0.9152
# Epoch 38/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.0754 - acc: 0.9732 - val_loss: 0.3624 - val_acc: 0.8937
# Epoch 39/40
# 163/163 [==============================] - 247s 2s/step - loss: 0.0743 - acc: 0.9728 - val_loss: 0.3023 - val_acc: 0.9074
# Epoch 40/40
# 163/163 [==============================] - 246s 2s/step - loss: 0.0660 - acc: 0.9760 - val_loss: 0.4522 - val_acc: 0.8701