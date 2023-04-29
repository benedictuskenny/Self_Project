#!/usr/bin/env python
# coding: utf-8

# ## 1. Library

import io
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


TRAIN_SIZE = 0.9
VOCAB_SIZE = 2000
MAX_LENGTH = 25
EMBEDDING_DIM = 16
TRUNC_TYPE = 'pre'
PADDING_TYPE = 'pre'
OOV_TOKEN = '<OOV>'
LSTM_DIM = 64


# ## 2. Data

data = pd.read_csv('fake_or_real_news.csv')
data = data[['title', 'label']]
data['label'] = data['label'].map({'REAL':0, 'FAKE':1})

data.head()

data.dtypes

text = " ".join(i for i in data['title'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color='white').generate(text)
plt.figure(figsize=(40,20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

def train_test_split(data, train_size):
    m, n = data.shape
    random.seed(42)
    shuffle_index = random.sample(range(m),m)
    threshold = int(train_size*m)
    train_index = shuffle_index[:threshold]
    test_index = shuffle_index[threshold:]
    training_data = data.loc[train_index]
    testing_data = data.loc[test_index]
    
    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []
    
    for i in training_data['title']:
        training_sentences.append(i)
    for i in training_data['label']:
        training_labels.append(i)
    for i in testing_data['title']:
        testing_sentences.append(i)
    for i in testing_data['label']:
        testing_labels.append(i)
        
    training_sentences = np.array(training_sentences)
    training_labels = np.array(training_labels)
    testing_sentences = np.array(testing_sentences)
    testing_labels = np.array(testing_labels)
    return training_sentences, training_labels, testing_sentences, testing_labels

training_sentences, training_labels, testing_sentences, testing_labels = train_test_split(data, TRAIN_SIZE)

print(f'Training headlines shape: {training_sentences.shape}')
print(f'Training labels shape: {training_labels.shape}')
print(f'Testing headlines shape: {testing_sentences.shape}')
print(f'Testing labels shape: {testing_labels.shape}')

print('Sample of training data:\n')
print(f'Headlines: {training_sentences[0]}\n')
print(f'Labeled as {training_labels[0]}\n')

print('Sample of testing data:\n')
print(f'Headlines: {testing_sentences[0]}\n')
print(f'Labeled as {testing_labels[0]}\n')


# ## 3. Using Naive Bayes

cv = CountVectorizer()
training_sentences = cv.fit_transform(training_sentences)
testing_sentences = cv.transform(testing_sentences)

model = MultinomialNB()
model.fit(training_sentences, training_labels)
print(model.score(testing_sentences, testing_labels))

def predict(headline):
    headline = cv.transform([headline])
    pred = model.predict(headline)
    pred = 'Fake' if pred==1 else "Real"
    return pred

# News: https://www.nytimes.com/2023/04/28/us/politics/pence-2024-campaign-trump.html
news_headline = "Campaign, Interrupted: Pence May Run, but He Can’t Hide From Trump’s Legal Woes"
predict(news_headline)

# News: https://www.nytimes.com/2023/04/28/us/politics/pence-2024-campaign-trump.html
news_headline = "Federal appeals court upholds Florida voting law that judge found discriminatory"
predict(news_headline)


# Note that "fake" means that it is fake news. Not a fake news headline.
# 
# This is news that proves the information is fake:
# In a split 2-1 decision, a panel of judges at the 11th U.S. Circuit Court of Appeals said the evidence did not show that lawmakers deliberately targeted Black voters when they passed provisions limiting the use of ballot drop boxes, barring third-party organizations from collecting voter registration forms and preventing people from engaging with voters in line.

# ## 4. Using NLP

training_sentences, training_labels, testing_sentences, testing_labels = train_test_split(data, TRAIN_SIZE)

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

model_nn = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_nn.summary()

model_nn_2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_nn_2.summary()

initial_weights_1 = model_nn.get_weights()
initial_weights_2 = model_nn_2.get_weights()


# ##### 1. Model 1 With Adam Optimizer and Binary Crossentropy Loss

train_model = model_nn
train_model.set_weights(initial_weights_1)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.Adam()
    
train_model.compile(loss='binary_crossentropy',
              optimizer=optimizer, 
              metrics=["accuracy"]) 

history_1 = train_model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[lr_schedule])

acc = history_1.history['accuracy']
val_acc = history_1.history['val_accuracy']
learning_rate = history_1.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# ##### 2. Model 1 With Adam Optimizer and Huber Loss

train_model = model_nn
train_model.set_weights(initial_weights_1)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.Adam()
    
train_model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer, 
              metrics=["accuracy"]) 

history_2 = train_model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[lr_schedule])

acc = history_2.history['accuracy']
val_acc = history_2.history['val_accuracy']
learning_rate = history_2.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# ##### 3. Model 1 With SGD Optimizer and Binary Crossentropy Loss

train_model = model_nn
train_model.set_weights(initial_weights_1)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    
train_model.compile(loss='binary_crossentropy',
              optimizer=optimizer, 
              metrics=["accuracy"]) 

history_3 = train_model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[lr_schedule])

acc = history_3.history['accuracy']
val_acc = history_3.history['val_accuracy']
learning_rate = history_3.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# ##### 4. Model 1 With SGD Optimizer and Huber Loss

train_model = model_nn
train_model.set_weights(initial_weights_1)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    
train_model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer, 
              metrics=["accuracy"]) 

history_4 = train_model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[lr_schedule])

acc = history_4.history['accuracy']
val_acc = history_4.history['val_accuracy']
learning_rate = history_4.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# ##### 5. Model 2 With Adam Optimizer and Binary Crossentropy Loss

train_model = model_nn_2
train_model.set_weights(initial_weights_2)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.Adam()
    
train_model.compile(loss='binary_crossentropy',
              optimizer=optimizer, 
              metrics=["accuracy"]) 

history_5 = train_model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[lr_schedule])

acc = history_5.history['accuracy']
val_acc = history_5.history['val_accuracy']
learning_rate = history_5.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# ##### 6. Model 2 With Adam Optimizer and Huber Loss

train_model = model_nn_2
train_model.set_weights(initial_weights_2)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.Adam()
    
train_model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer, 
              metrics=["accuracy"]) 

history_6 = train_model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[lr_schedule])

acc = history_6.history['accuracy']
val_acc = history_6.history['val_accuracy']
learning_rate = history_6.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# ##### 7. Model 2 With SGD Optimizer and Binary Crossentropy Loss

train_model = model_nn_2
train_model.set_weights(initial_weights_2)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    
train_model.compile(loss='binary_crossentropy',
              optimizer=optimizer, 
              metrics=["accuracy"]) 

history_7 = train_model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[lr_schedule])

acc = history_7.history['accuracy']
val_acc = history_7.history['val_accuracy']
learning_rate = history_7.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# ##### 8. Model 2 With SGD Optimizer and Huber Loss

train_model = model_nn_2
train_model.set_weights(initial_weights_2)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    
train_model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer, 
              metrics=["accuracy"]) 

history_8 = train_model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[lr_schedule])

acc = history_8.history['accuracy']
val_acc = history_8.history['val_accuracy']
learning_rate = history_8.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# #### Learning rate for model 1 with Adam optimizer and Binary Crossentropy loss

acc = history_1.history['accuracy']
val_acc = history_1.history['val_accuracy']
learning_rate = history_1.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.axis([0, 0.01, 0.5, 1])
plt.show()


# #### Learning rate for model 1 with Adam optimizer and Huber loss

acc = history_2.history['accuracy']
val_acc = history_2.history['val_accuracy']
learning_rate = history_2.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.axis([0, 0.006, 0.5, 1])
plt.show()


# #### Learning rate for model 2 with Adam optimizer and Binary Crossentropy loss

acc = history_5.history['accuracy']
val_acc = history_5.history['val_accuracy']
learning_rate = history_5.history['lr']

plt.plot(learning_rate, acc, 'red')
plt.plot(learning_rate, val_acc, 'blue')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.axis([0, 0.005, 0.5, 1])
plt.show()


# The best combination for this case are:
# 1. Model 1 with Adam Optimizer and Binary Crossentropy Loss (with a learning rate of about 0.005)
# 2. Model 1 with Adam Optimizer and Huber Loss (with a learning rate of about 0.003)
# 3. Model 2 with Adam Optimizer and Binary Crossentropy Loss (with a learning rate of about 0.002)

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= 0.92):
            print("\nReached 92% accuracy so cancelling training!")
            self.model.stop_training = True        
callbacks = mycallback()


# #### Model 1 with Adam optimizer and Binary Crossentropy loss (with learning_rate = 0.002)

alt_model_1 = model_nn
alt_model_1.set_weights(initial_weights_1)

alt_model_1.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), 
              metrics=["accuracy"]) 

train_history_1 = alt_model_1.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[callbacks])

acc = train_history_1.history['accuracy']
val_acc = train_history_1.history['val_accuracy']
number_epochs = range(len(acc))

plt.style.use("bmh")
plt.figure( figsize=(12,8))
plt.plot(number_epochs, acc)
plt.plot(number_epochs, val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training VS Validation Accuracy Model 1')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()

# #### Model 1 with Adam optimizer and huber loss (with learning_rate = 0.003)

alt_model_2 = model_nn
alt_model_2.set_weights(initial_weights_1)

alt_model_2.compile(loss=tf.keras.losses.Huber(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), 
              metrics=["accuracy"]) 

train_history_2 = alt_model_2.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[callbacks])

acc = train_history_2.history['accuracy']
val_acc = train_history_2.history['val_accuracy']
number_epochs = range(len(acc))

plt.figure( figsize=(12,8))
plt.plot(number_epochs, acc, 'red')
plt.plot(number_epochs, val_acc, 'blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training VS Validation Accuracy Model 2')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# #### Model 2 with Adam optimizer and Binary Crossentropy loss (with learning_rate=0.002)

alt_model_3 = model_nn_2
alt_model_3.set_weights(initial_weights_2)

alt_model_3.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), 
              metrics=["accuracy"]) 

train_history_3 = alt_model_3.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), callbacks=[callbacks])

acc = train_history_3.history['accuracy']
val_acc = train_history_3.history['val_accuracy']
number_epochs = range(len(acc))

plt.figure( figsize=(12,8))
plt.plot(number_epochs, acc, 'red')
plt.plot(number_epochs, val_acc, 'blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training VS Validation Accuracy Model 3')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()


# #### Predict

def predict_nn(headline, threshold):
    sequence = tokenizer.texts_to_sequences([headline])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    
    prob1 = round(float(alt_model_1.predict(padded)),4)
    prob2 = round(float(alt_model_2.predict(padded)),4)
    prob3 = round(float(alt_model_3.predict(padded)),4)
    res1 = 'FAKE' if prob1 >= threshold else 'REAL'
    res2 = 'FAKE' if prob2 >= threshold else 'REAL'
    res3 = 'FAKE' if prob3 >= threshold else 'REAL'
    
    print(f'First model result is "{res1}"')
    print(f'Second model result is "{res2}"')
    print(f'Third model result is "{res3}"')

# News: https://www.nytimes.com/2023/04/28/us/politics/pence-2024-campaign-trump.html
news_headline = "Campaign, Interrupted: Pence May Run, but He Can’t Hide From Trump’s Legal Woes"
predict_nn(news_headline, 0.2)

# News: https://www.nytimes.com/2023/04/28/us/politics/pence-2024-campaign-trump.html
news_headline = "Federal appeals court upholds Florida voting law that judge found discriminatory"
predict_nn(news_headline, 0.2)


# ##### Note that "fake" means that it is fake news. Not a fake news headline.
# 
# This is news that proves the information is fake:
# 
# In a split 2-1 decision, a panel of judges at the 11th U.S. Circuit Court of Appeals said the evidence did not show that lawmakers deliberately targeted Black voters when they passed provisions limiting the use of ballot drop boxes, barring third-party organizations from collecting voter registration forms and preventing people from engaging with voters in line.