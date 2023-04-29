# Fake News Detection

One of the most harmful aspects of social media applications is the spreading of fake news. That's why machine learning can be used to detect whether the news is fake or real based on the headline. You can download the dataset here: https://www.kaggle.com/hassanamin/textdb3/download. In this project, I used basic machine learning with the Naive Bayes algorithm and NLP. 

### Word Cloud
![image](https://user-images.githubusercontent.com/125811483/235296050-d37afad3-6369-4c39-b85a-9d3b54c93d53.png)


### News that I used to predict
* https://www.nytimes.com/2023/04/28/us/politics/pence-2024-campaign-trump.html
* https://www.nytimes.com/2023/04/28/us/politics/pence-2024-campaign-trump.html

### Naive Bayes Algorithm result
![image](https://user-images.githubusercontent.com/125811483/235295224-d0948f17-1f48-4a6f-9f1f-7ca77fa0ca89.png)

### Natural Language Processing (NLP) result
In the NLP section, I created two models. The first neural network model consists of embeddings, bidirectional LSTM, dropout, and three dense layers. For the second neural network model, there is an additional layer of bidirectional LSTM. For the loss function, I chose to use binary cross-entropy and Huber loss. For the optimizer, I chose to use Adam and Stochastic Gradient Descent (SGD). Because there are a lot of combinations that can be made, I decided to create all eight combinations. In the end, there are three best combinations, which are:

1. Model 1 with Adam Optimizer and Binary Crossentropy Loss (with a learning rate of about 0.005)
2. Model 1 with Adam Optimizer and Huber Loss (with a learning rate of about 0.003)
3. Model 2 with Adam Optimizer and Binary Crossentropy Loss (with a learning rate of about 0.002)

Then, these three models will be trained until their accuracy reaches 92% using callbacks. These are the results.

![image](https://user-images.githubusercontent.com/125811483/235296064-ce828ada-5382-4441-8135-8fc7660df4c4.png)

As we can see, these three models almost have the same accuracy. But all of them indicate overfitting.

This is the final result for detecting fake news from the news headline:
![image](https://user-images.githubusercontent.com/125811483/235295907-95d701ef-63b2-48d5-afcc-7666d7312bb3.png)
![image](https://user-images.githubusercontent.com/125811483/235296084-37222ecb-1c72-4bf3-867a-db59d2ed1031.png)
![image](https://user-images.githubusercontent.com/125811483/235296088-a7566cc1-5c28-41f2-84ef-7be2358d1075.png)


### Additional Information
Anyway, there is something interesting if you follow these steps :D

1. Download "fake_news_vecs.tsv" and "fake_news_meta.tsv"

2. Go to https://projector.tensorflow.org/

3. Click "load" on the left side.

4. Upload "fake_news_vecs.tsv" at Load a TSV file of vectors.

5. Upload "fake_news_meta.tsv" at Load a TSV file of metadata.

6. Click anywhere to close it.

7. Voila! You can search for your desired words (i.e., politics and California) with the search feature on the right side.
