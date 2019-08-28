# https://www.kaggle.com/antoange/chest-x-ray-pneumonia-detection-with-keras-cnn
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


print('Teach CNN? [y/n]')
# input
todo1 = input()

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

    training_set = train_datagen.flow_from_directory('./chest-xray-pneumonia/chest_xray/train',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('./chest-xray-pneumonia/chest_xray/test',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    classifier.summary()

    print('Running fit_generator')
    print('You can go for coffee. Or for lunch. Or to smoke. Or all of this, you will have a lot of time now...')

    history = classifier.fit_generator(training_set,
                            steps_per_epoch = 163,
                            epochs = 10,
                            validation_data = test_set,
                            validation_steps = 624)


    print('Save the model? [y/n]')
    # input
    todo2 = input()

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
    plt.show()

    #Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Test set'], loc='upper left')
    plt.show()

print('Load the model? [y/n]')
# input
todo3 = input()

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


print('Test the model? [y/n]')
# input
todo4 = input()
if (todo4 == 'y'):
    # break
# else:
    
    # predicting images
    # TODO: use ImageDataGenerator
    # img1 = image.load_img('./chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg', target_size=(64, 64))
    # img2 = image.load_img('./chest-xray-pneumonia/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg', target_size=(64, 64))
    
    # x = image.img_to_array(img1)
    # x = np.expand_dims(x, axis=0)

    # images = np.vstack([x])
    # classes = classifier.predict_classes(images, batch_size=10)
    # print(classes)

    # x = image.img_to_array(img2)
    # x = np.expand_dims(x, axis=0)

    # images = np.vstack([x])
    # classes = classifier.predict_classes(images, batch_size=10)
    # print(classes)

    img = image.load_img('./chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("PNEUMONIA =", preds)

    img = image.load_img('./chest-xray-pneumonia/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("NORMAL =", preds)

    img = image.load_img('./chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1952_bacteria_4883.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("PNEUMONIA =", preds)

    img = image.load_img('./chest-xray-pneumonia/chest_xray/val/NORMAL/NORMAL2-IM-1431-0001.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("NORMAL =", preds)

    img = image.load_img('./chest-xray-pneumonia/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg', target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = classifier.predict(x)
    print("PNEUMONIA =", preds)




# (NN_Env) C:\Users\ftokarev\Documents\AI University\task1\pulsaring\aiuniversityhomework>python pneumoniacheck.py
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
# _________________________________________________________________

# (NN_Env) C:\Users\ftokarev\Documents\AI University\task1\pulsaring\aiuniversityhomework>python pneumoniacheck.py
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
# _________________________________________________________________
# WARNING:tensorflow:From C:\Users\ftokarev\AppData\Local\Continuum\anaconda3\envs\NN_Env\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.cast instead.
# Epoch 1/10
# 2019-08-26 16:43:31.634141: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
# 163/163 [==============================] - 655s 4s/step - loss: 0.3563 - acc: 0.8407 - val_loss: 0.3026 - val_acc: 0.8784
# Epoch 2/10
# 163/163 [==============================] - 479s 3s/step - loss: 0.2258 - acc: 0.9049 - val_loss: 0.4853 - val_acc: 0.7933
# Epoch 3/10
# 163/163 [==============================] - 479s 3s/step - loss: 0.1989 - acc: 0.9212 - val_loss: 0.2789 - val_acc: 0.8893
# Epoch 4/10
# 163/163 [==============================] - 480s 3s/step - loss: 0.2075 - acc: 0.9189 - val_loss: 0.4403 - val_acc: 0.8353
# Epoch 5/10
# 163/163 [==============================] - 1076s 7s/step - loss: 0.1673 - acc: 0.9356 - val_loss: 0.2661 - val_acc: 0.8990
# Epoch 6/10
# 163/163 [==============================] - 728s 4s/step - loss: 0.1635 - acc: 0.9367 - val_loss: 0.2696 - val_acc: 0.8909
# Epoch 7/10
# 163/163 [==============================] - 666s 4s/step - loss: 0.1631 - acc: 0.9367 - val_loss: 0.3588 - val_acc: 0.8796
# Epoch 8/10
# 163/163 [==============================] - 673s 4s/step - loss: 0.1449 - acc: 0.9431 - val_loss: 0.3136 - val_acc: 0.8910
# Epoch 9/10
# 163/163 [==============================] - 530s 3s/step - loss: 0.1406 - acc: 0.9463 - val_loss: 0.3288 - val_acc: 0.8832
# Epoch 10/10
# 163/163 [==============================] - 515s 3s/step - loss: 0.1393 - acc: 0.9477 - val_loss: 0.5068 - val_acc: 0.8399
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

# (NN_Env) C:\Users\ftokarev\Documents\AI University\task1\pulsaring\aiuniversityhomework>
