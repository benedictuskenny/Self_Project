# HUMAN FACE DETECTION

Face detection is a computer technique that recognizes human faces in digital pictures and is used in a range of applications. Face detection is also the psychological process through which humans find and attend to faces in a visual context. Face detection could be considered a subset of object-class detection. The goal of object-class detection is to determine the positions and sizes of all objects in a picture that correspond to a specific class. This data project can be found at https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection. This data has 2204 images and 1 CSV file. The CSV file contains the file name and the (x,y) coordinates where the face is. Note that this model might not be the best.

### Data

This is what the last five images look like.
![image](https://user-images.githubusercontent.com/125811483/235614861-3efde3bc-9f83-475a-a2a3-4e16a4f8fee3.png)

### Models

For the model, I created two CNN models with a binary cross-entropy loss function and an Adam optimizer.

The first model consists of:

* 3 convolutional layers

* 3 dense layers

* Also with max-pooling and flattening layers

The first model consists of:

* 4 convolutional layers

* 3 dense layers

* Some dropout and batch normalization layers

* Also with max-pooling and flattening layers



### Result

This is the training and validation loss plot from the first model.

![image](https://user-images.githubusercontent.com/125811483/235615057-1efe02e9-799d-495e-b488-9b4c333409e4.png)

And this is the training and validation loss plot from the second model.

![image](https://user-images.githubusercontent.com/125811483/235615066-5a5ce22e-233c-433e-8503-d6189764079f.png)

As a result, there is overfitting for both the first and second models. For prediction, both models have their own performance. Sometimes both of them have good performances, but sometimes only one of them can detect the face well. These are examples (the left one is predicted using model 1, and the right one is predicted using model 2). 

* Both models have good performances.
  * First example

  ![image](https://user-images.githubusercontent.com/125811483/235616320-8814e6d7-cb64-4d05-b32c-558f70cbe62e.png)
  ![image](https://user-images.githubusercontent.com/125811483/235616333-d08f554a-bcf8-43b7-b759-c28edc7ae052.png)

  * Second example
  
  ![image](https://user-images.githubusercontent.com/125811483/235616356-399bcd5f-5a99-4086-8611-44a4b702c6c1.png)
  ![image](https://user-images.githubusercontent.com/125811483/235616377-4a20fcde-41b3-45b5-b279-5d8a12d63c33.png)


* Model 1 is better.
  * First example
  
  ![image](https://user-images.githubusercontent.com/125811483/235616733-e5d0a8d8-34cb-424c-a1db-32a906515f47.png)
  ![image](https://user-images.githubusercontent.com/125811483/235616753-0c620d1e-f60b-4ec8-bd55-c80b4890d853.png)

  * Second example (Both models are good, but the first model is slightly better)
  
  ![image](https://user-images.githubusercontent.com/125811483/235616800-f7e28763-9cd5-4498-b751-0f6e3973a243.png)
  ![image](https://user-images.githubusercontent.com/125811483/235616818-2d56a957-1f95-4ed1-8ffc-4af36805ce42.png)


* Model 2 is better.
  * First example
  
  ![image](https://user-images.githubusercontent.com/125811483/235616602-a0314770-f27e-49c9-ab25-924d8c1528b6.png)
  ![image](https://user-images.githubusercontent.com/125811483/235616615-b29a91a8-6db8-40f9-94bd-6ad966ab547e.png)

  * Second example
  
  ![image](https://user-images.githubusercontent.com/125811483/235616461-ff1ceddf-fde4-4dbf-817b-2edc8fc77dc2.png)
  ![image](https://user-images.githubusercontent.com/125811483/235616484-6ec6835d-2d49-4279-920e-ebcdd8f5d7a9.png)

* Both of them failed.
Both of the models seem to have failed in the pictures that contain more than one face (multiple faces).
![image](https://user-images.githubusercontent.com/125811483/235616394-ea88905e-c89e-40d9-8c1b-10f82b947fdc.png)
![image](https://user-images.githubusercontent.com/125811483/235616414-48ab0bf0-c0ef-4196-a174-f83d3e7a3d38.png)

### That's it!

Any advices or recommendations would be much appreciated for my evaluation. Thanks in advance!
