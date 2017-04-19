import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # importing data from center, left and right cameras
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    filename = source_path.split('\\')[-1]
                    local_path = './data/IMG/' + filename
                    image = cv2.imread(local_path)
                    # convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                correction = 0.2
                measurement = float(line[3])
                angles.append(measurement)
                angles.append(measurement+correction)
                angles.append(measurement-correction)

            # generates a flipped view of data
            augmented_images = []
            augmented_angles =[]
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                flipped_images = cv2.flip(image, 1)
                flipped_angles = float(angle) * -1.0
                augmented_images.append(flipped_images)
                augmented_angles.append(flipped_angles)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

# nvidia network architecture
def nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

# commaai train steering model
def commaai():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((65,25), (0,0))))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

model = nvidia()

history_object = model.fit_generator(train_generator,
    steps_per_epoch = len(train_samples)/32,
    validation_data = validation_generator,
    validation_steps = len(validation_samples)/32,
    nb_epoch=4, verbose=1)
print(history_object.history.keys())

model.save('model.h5')
print('model saved')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./examples/loss.png')
