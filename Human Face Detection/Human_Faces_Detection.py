import os
import cv2 as cv
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import mean_absolute_error

! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d sbaghbidi/human-faces-object-detection

!unzip human-faces-object-detection.zip

df = pd.read_csv('faces.csv')

df.head(7)

df.shape

"""# 2. Preparation

Resizing the images
"""

data = {}
for i in df['image_name']:
  if i not in data:
    data[i] = []

for idx, img_name in enumerate(df['image_name']):
  width = df["width"][idx]
  height = df["height"][idx]
  x1 = df["x0"][idx]
  y1 = df["y0"][idx]
  x2 = df["x1"][idx]
  y2 = df["y1"][idx]
  new_x1 = int((x1/width)*128)
  new_y1 = int((y1/height)*128)
  new_x2 = int((x2/width)*128)
  new_y2 = int((y2/height)*128)
  data[img_name].append(new_x1)
  data[img_name].append(new_y1)
  data[img_name].append(new_x2)
  data[img_name].append(new_y2)

image_dir = os.listdir('/content/images')
images = []
for image_name in data.keys():
  for itr in image_dir:
    if image_name == itr:
      image_arr = cv.imread(os.path.join('/content/images', image_name), cv.IMREAD_GRAYSCALE)
      resized_image = cv.resize(image_arr, (128, 128))
      images.append(resized_image)

images = np.array(images)
images = np.expand_dims(images, axis=3)

print(f"Images shape: {images.shape}")

"""Bounding Box"""

bbox = []
for boxes in data.keys():
  bbox.append(data[boxes])

maxlen = 0
for i in bbox:
  length = len(i)
  if length > maxlen:
    maxlen = length
bbox = tf.keras.preprocessing.sequence.pad_sequences(bbox, maxlen=max, padding='post')

bbox = np.array(bbox)
print(f"shape of bbox {bbox.shape}")

"""Rescale"""

images = images/255
bbox = bbox/128

plt.figure(figsize=(15,10))
for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.imshow(images[-i], cmap='gray')
  plt.axis("off")

def split_data(X, Y, train_size):
  m, n = Y.shape
  random.seed(42)
  shuffle_index = random.sample(range(m), m)
  train_index = int(m*train_size)

  data_training = X[:train_index]
  data_testing = X[train_index:]
  label_training = Y[:train_index]
  label_testing = Y[train_index:]

  return data_training, label_training, data_testing, label_testing

data_training, label_training, data_testing, label_testing = split_data(images, bbox, 0.85)

print(f"Training images shape: {data_training.shape}")
print(f"Training labels shape: {label_training.shape}")
print(f"Testing images shape: {data_testing.shape}")
print(f"Testing labels shape: {label_testing.shape}")

"""# 3. CNN Model"""

model1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(48, activation='sigmoid')
])
model1_initial_weights = model1.get_weights()
model1.summary()

model1.set_weights(model1_initial_weight)

model1.compile(optimizer = "adam", 
               loss="binary_crossentropy")

history1 = model1.fit(data_training, label_training, validation_data=(data_testing, label_testing), epochs=25)

loss = history1.history['loss']
val_loss = history1.history['val_loss']
epoch = range(len(loss))

plt.plot(epoch, loss)
plt.plot(epoch, val_loss)
plt.legend(["Training Loss", "Validation Loss"])
plt.show()

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', input_shape=(128,128,1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(48, activation='sigmoid')
])
model2_initial_weights = model2.get_weights()
model2.summary()

model2.set_weights(model2_initial_weights)

model2.compile(optimizer = "adam", 
               loss="binary_crossentropy")

history2 = model2.fit(data_training, label_training, validation_data=(data_testing, label_testing), epochs=30)

loss = history2.history['loss']
val_loss = history2.history['val_loss']
epoch = range(len(loss))

plt.plot(epoch, loss)
plt.plot(epoch, val_loss)
plt.legend(["Training Loss", "Validation Loss"])
plt.show()

"""Trained Weights"""

model1_trained_weights = model1.get_weights()
model2_trained_weights = model2.get_weights()

"""# 4. Prediction"""

def model_prediction(num, model):
  data = data_testing[num]
  prediction = model.predict(data.reshape(1, 128, 128, 1))
  fig, ax = plt.subplots(1)
  ax.imshow(data)
  x1 = int(prediction[0][0]*128)
  y1 = int(prediction[0][1]*128)
  x2 = int(prediction[0][2]*128)
  y2 = int(prediction[0][3]*128)
  rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='red', facecolor="none")
  ax.add_patch(rect)
  plt.show()

def model_prediction(num, model):
  for i in range(num):
    data = data_testing[i]
    prediction = model.predict(data.reshape(1, 128, 128, 1))
    num_sqrt = int(np.sqrt(num))
    fig, ax = plt.subplot(num_sqrt, num_sqrt, i+1)
    ax.imshow(data)
    x1 = int(prediction[0][0]*128)
    y1 = int(prediction[0][1]*128)
    x2 = int(prediction[0][2]*128)
    y2 = int(prediction[0][3]*128)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='red', facecolor="none")
    ax.add_patch(rect)
    plt.show()

num = 75 # The first model is better
model_prediction(num, model1)
model_prediction(num, model2)

num = 15# The second model is better
model_prediction(num, model1)
model_prediction(num, model2)

num = 91 # Both models are good, but the first model is slightly better
model_prediction(num, model1)
model_prediction(num, model2)

num = 7 # The second model is better
model_prediction(num, model1)
model_prediction(num, model2)

num = 11 # Both models are good
model_prediction(num, model1)
model_prediction(num, model2)

num = 18 # Both models are good
model_prediction(num, model1)
model_prediction(num, model2)

num = 10 # Both models failed.
model_prediction(num, model1)
model_prediction(num, model2)
