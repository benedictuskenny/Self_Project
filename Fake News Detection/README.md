# Fake News Detection

One of the most harmful aspects of social media applications is the spreading of fake news. That's why machine learning can be used to detect whether the news is fake or real based on the headline. You can download the dataset here: https://www.kaggle.com/hassanamin/textdb3/download. In this project, I used basic machine learning with the Naive Bayes algorithm and NLP. 

# Word Cloud
![image](https://user-images.githubusercontent.com/125811483/235296395-1482d24a-16dc-4cb1-858f-0a185a07e8b9.png)


# News that I used to predict
* https://www.nytimes.com/2023/04/28/us/politics/pence-2024-campaign-trump.html
* https://www.nytimes.com/2023/04/28/us/politics/pence-2024-campaign-trump.html

Note that "fake" means that it is fake news. Not a fake news headline.

# Naive Bayes Algorithm result
![image](https://user-images.githubusercontent.com/125811483/235295224-d0948f17-1f48-4a6f-9f1f-7ca77fa0ca89.png)

# Natural Language Processing (NLP) result
In the NLP section, I created two models. The first neural network model consists of embeddings, bidirectional LSTM, dropout, and three dense layers. For the second neural network model, there is an additional layer of bidirectional LSTM. For the loss function, I chose to use binary cross-entropy and Huber loss. For the optimizer, I chose to use Adam and Stochastic Gradient Descent (SGD). Because there are a lot of combinations that can be made, I decided to create all eight combinations. In the end, there are three best combinations, which are:

1. Model 1 with Adam Optimizer and Binary Crossentropy Loss (with a learning rate of about 0.005)
2. Model 1 with Adam Optimizer and Huber Loss (with a learning rate of about 0.003)
3. Model 2 with Adam Optimizer and Binary Crossentropy Loss (with a learning rate of about 0.002)

Then, these three models will be trained until their accuracy reaches 92% using callbacks. These are the results.
![image](https://user-images.githubusercontent.com/125811483/235296681-8a54c3e3-b574-41d4-94ce-b70eddd7f4bd.png)
![image](https://user-images.githubusercontent.com/125811483/235296686-be2effd5-7533-4d0e-937c-b5b40041ce01.png)
![image](https://user-images.githubusercontent.com/125811483/235296706-0bb8f580-c48f-40dc-8f1e-34dead8b02a1.png)



As we can see, these three models almost have the same accuracy. But all of them indicate overfitting.

This is the final result for detecting fake news from the news headline:
![image](https://user-images.githubusercontent.com/125811483/235295907-95d701ef-63b2-48d5-afcc-7666d7312bb3.png)

# Additional Information
Anyway, there is something interesting if you follow these steps :D

1. Download "fake_news_vecs.tsv" and "fake_news_meta.tsv"

2. Go to https://projector.tensorflow.org/

3. Click "load" on the left side.

4. Upload "fake_news_vecs.tsv" at Load a TSV file of vectors.

5. Upload "fake_news_meta.tsv" at Load a TSV file of metadata.

6. Click anywhere to close it.

7. Voila! You can search for your desired words (i.e., politics and California) with the search feature on the right side.

# That's it!

Any advices or recommendations would be much appreciated for my evaluation. Thanks in advance!
