import keras
import tensorflow as tf
from sklearn.datasets import load_files
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt

# Data specifications
# This script should be in the overarching directory of the fruit360 dataset. 

train_dir = "fruit360/fruits-360_dataset/fruits-360/Training"
test_dir = "fruit360/fruits-360_dataset/fruits-360/Test"

# Load dataset method

def load_dataset(path):
    data = load_files(path)
    files = np.array(data["filenames"])
    targets = np.array(data["target"])
    target_labels = np.array(data["target_names"])
    return files,targets,target_labels
    

# Load data 

x_train, y_train,target_labels = load_dataset(train_dir)
x_test, y_test,_ = load_dataset(test_dir)
print("Loading complete!")

print("Training set size : " + str(x_train.shape[0]))
print("Testing set size : " + str(x_test.shape[0]))

# Let's confirm the number of classes

no_of_classes = len(np.unique(y_train))
print("Number of classes / fruits: " + str(no_of_classes))

# The target labels are originally numbers corresponding to class labels

print("Original labels" + str(y_train[0:10]))

# Make one-hot

y_train = np_utils.to_categorical(y_train,no_of_classes)
y_test = np_utils.to_categorical(y_test,no_of_classes)
y_train[0] 

# Divide the test samples into validation and test samples

x_test,x_valid = x_test[6969:],x_test[:6969]
y_test,y_valid = y_test[6969:],y_test[:6969]
print("Size of x_valid: " + str(x_valid.shape))
print("Size of y_valid: " + str(y_valid.shape))
print("Size of x_test: " + str(x_test.shape))
print("Size of y_test: " + str(y_test.shape))

# Convert Images to np arrays

def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(file)))
    return images_as_array

x_train = np.array(convert_image_to_array(x_train))
print("Training set shape: " + str(x_train.shape))

x_valid = np.array(convert_image_to_array(x_valid))
print("Validation set shap: " + str(x_valid.shape))

x_test = np.array(convert_image_to_array(x_test))
print("Test set shape: " + str(x_test.shape))

print("Training image shape: " + str(x_train[0].shape))

# Convert pixel values to 0->1 

x_train = x_train.astype('float32')/255
x_valid = x_valid.astype('float32')/255
x_test = x_test.astype('float32')/255

# Plot images

fig = plt.figure(figsize =(28,6))
for i in range(12):
    ax = fig.add_subplot(3,4,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(x_train[i]))

plt.savefig("fruits.png")
plt.show()
print("Done plotting")

# Creating the CNN

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K

# Add 3 convolutional layers and then two fully connected ones

model = Sequential()
model.add(Conv2D(filters=32, kernel_size = (4,4), input_shape = (100, 100, 3), activation="relu", strides=2, padding="valid"))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(filters=64, kernel_size=2, padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=2))

# Move to fully connected layers

model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(270, use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(200, use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(131, activation='softmax'))
print(model.summary())

# Use Adam optimization

model.compile(loss=tf.keras.losses.CategoricalCrossentropy() , optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
print('Compiled!')

# Metadata and model fit

batch_size = 32
checkpointer = ModelCheckpoint(filepath = 'cnn_fruits.hdf5', verbose = 1, save_best_only = True)
history = model.fit(x_train,y_train, batch_size = 32, epochs=5, validation_data=(x_valid, y_valid), callbacks = [checkpointer], verbose=2, shuffle=True)

# Load the weights that yielded the best validation accuracy

model.load_weights('cnn_fruits.hdf5')

# Evaluate and print test accuracy

score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

# Visualize test prediction.

y_pred = model.predict(x_test)

# Plot a random sample of test images, the model's predicted labels, and ground truth

fig = plt.figure(figsize=(16, 9))

for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))

plt.savefig("fruits_model.png")
plt.show()
plt.figure(1)  
   

