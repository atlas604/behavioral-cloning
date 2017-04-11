import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

lines = []
with open('./test/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for lines in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        filename = source_path.split('\\')[-1]
        local_path = './test/IMG/' + filename
        image = cv2.imread(local_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    correction = 0.2
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

augmented_images = []
augmented_measurements =[]
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_images = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0
    augmented_images.append(flipped_images)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# nvidia model with dropout implementation
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
print('model saved')

plt.hist(y_train, bins=50)
plt.savefig('hist.png')
print('done')
