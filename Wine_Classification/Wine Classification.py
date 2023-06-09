# -*- coding: utf-8 -*-
"""Copy of Wine_Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QQzulbSFdbeI2W_wuNE4UDygw_MoEsMX
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

"""# Prepare the Dataset"""

from google.colab import drive
drive.mount('/content/drive')

dataframe = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Self-Project/Wine/wine.csv')
dataframe.head()

dataframe.tail()

dataframe.dtypes

N = len(dataframe)
row, col = dataframe.shape
col = col-1

for k in range(N):
    if dataframe.iloc[k,col] == 'good':
        dataframe.iloc[k,col] = 1
    else:
        dataframe.iloc[k,col] = 0

dataframe.head()

dataframe["quality"] = pd.Series(dataframe['quality'], dtype='category')

sns.countplot(x=dataframe['quality'])

# Resampling using SMOTE
sm = SMOTE(random_state=42)
X = dataframe.iloc[:,:-1]
y = dataframe.iloc[:,-1]
X_bal, y_bal = sm.fit_resample(X,y)
sns.countplot(x=y_bal)

dataframe = pd.concat([X_bal, y_bal], axis=1)
dataframe

dataframe.isnull().sum()

fig = plt.gcf()
figsize = fig.get_size_inches()
fig.set_size_inches(figsize*2)
sns.heatmap(dataframe.corr(), annot = True)

"""# Basic Supervised Learning"""

predictor = ['volatile acidity', 
             'citric acid', 
             'chlorides', 
             'total sulfur dioxide', 
             'density', 
             'sulphates', 
             'alcohol']
target = ['quality']

X = dataframe[predictor]
Y = dataframe[target]

Xtrain, Xtest, Ytrain, Ytest = tts(X, Y, test_size = 0.3, random_state = 13)
Ytrain = pd.to_numeric(Ytrain.iloc[:,0])
Ytest = pd.to_numeric(Ytest.iloc[:,0])
Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape

from sklearn.linear_model import LogisticRegression

regression_model = LogisticRegression(max_iter = 500)
regression_model.fit(Xtrain, Ytrain)
Ypred = regression_model.predict(Xtest)

cm = confusion_matrix(Ytest, Ypred)
fig, ax = plt.subplots()
sns.set(font_scale = 1.5) 
sns.heatmap(cm, annot = True, cmap = 'Blues', fmt = 'g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('Actual labels')
ax.set_title('Logistic Regression')

print(classification_report(Ytest, Ypred))

from sklearn import tree

DTC = tree.DecisionTreeClassifier().fit(Xtrain,Ytrain)
Ypred = DTC.predict(Xtest)

cm = confusion_matrix(Ytest, Ypred)
fig, ax = plt.subplots()
sns.set(font_scale = 1.5) 
sns.heatmap(cm, annot = True, cmap = 'Reds', fmt = 'g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('Actual labels')
ax.set_title('Decision Tree Classifier (CART)')

print(classification_report(Ytest, Ypred))

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators = 100).fit(Xtrain, Ytrain)
Ypred = RFC.predict(Xtest)

cm = confusion_matrix(Ytest, Ypred)
fig, ax = plt.subplots()
sns.set(font_scale = 1.5)
sns.heatmap(cm, annot = True, cmap = 'Greens', fmt = 'g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('Actual labels')
ax.set_title('Random Forest')

print(classification_report(Ytest, Ypred))

from sklearn.ensemble import AdaBoostClassifier

ABC = AdaBoostClassifier(n_estimators = 100).fit(Xtrain, Ytrain)
Ypred = ABC.predict(Xtest)

cm = confusion_matrix(Ytest, Ypred)
fig, ax = plt.subplots()
sns.set(font_scale = 1.5)
sns.heatmap(cm, annot = True, cmap = 'Purples', fmt = 'g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('Actual labels')
ax.set_title('AdaBoost')

print(classification_report(Ytest, Ypred))

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(n_estimators = 100).fit(Xtrain, Ytrain)
Ypred = GBC.predict(Xtest)

cm = confusion_matrix(Ytest, Ypred)
fig, ax = plt.subplots()
sns.set(font_scale = 1.5)
sns.heatmap(cm, annot = True, cmap = 'Oranges', fmt = 'g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('Actual labels')
ax.set_title('Gradient Boosting')

print(classification_report(Ytest, Ypred))

import xgboost

XGB = xgboost.XGBClassifier(n_estimators = 1000).fit(Xtrain,Ytrain)
Ypred = XGB.predict(Xtest)

cm = confusion_matrix(Ytest, Ypred)
fig, ax = plt.subplots()
sns.set(font_scale = 1.5)
sns.heatmap(cm, annot = True, cmap = 'Oranges', fmt = 'g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('Actual labels')
ax.set_title('XGBoost')

print(classification_report(Ytest, Ypred))

"""# Neural Network"""

dataframe.head()

dataframe.shape

# Normalizing
def normalize(data):
  m, n = data.shape
  
  # Notes: The last column is the label or target
  name = data.columns[-1]
  label = data.iloc[:,-1]
  data = data.iloc[:,:-1]
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  
  data = (data-mean)/std
  data[name] = label
  return(data)

# Train and Test Split
def train_test_split(data, train_size):
  n = len(data)
  train_size = int(n*train_size)

  shuffle_index = random.sample(range(n), n)
  train_index = shuffle_index[:train_size]
  test_index = shuffle_index[train_size:]

  data_train = data.loc[train_index]
  data_test = data.loc[test_index]

  # Convert to tensor
  train_data = tf.convert_to_tensor(data_train.iloc[:,:-1])
  train_label = tf.convert_to_tensor(data_train.iloc[:,-1])
  test_data = tf.convert_to_tensor(data_test.iloc[:,:-1])
  test_label = tf.convert_to_tensor(data_test.iloc[:,-1])
  return train_data, train_label, test_data, test_label

train_data, train_label, validation_data, validation_label = train_test_split(normalize(dataframe), 0.9)
print('Train data shape: ', train_data.shape)
print('Train label shape: ', train_label.shape)
print('Validation data shape: ', validation_data.shape)
print('Validation label shape: ', validation_label.shape)

# Create model
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.90):
      print(" =========> MODEL ACCURACY REACHED 90%, STOP TRAINING!")
      self.model.stop_training = True

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(None, train_data.shape[1])),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

epoch = model.fit(x=train_data, y=train_label, epochs=50, callbacks=[myCallback()], validation_data=(validation_data, validation_label))

loss = epoch.history['loss']
acc = epoch.history['accuracy']
val_loss = epoch.history['val_loss']
val_acc = epoch.history['val_accuracy']


number_epochs=range(len(acc))

plt.plot(number_epochs, acc, 'red')
plt.plot(number_epochs, val_acc, 'blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training VS Validation Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()

plt.plot(number_epochs, loss, 'red')
plt.plot(number_epochs, val_loss, 'blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training VS Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()