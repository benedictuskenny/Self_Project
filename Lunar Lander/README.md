# Lunar Lander
‘Lunar Lander’ environment from the OpenAI gym This environment deals with the problem of landing a lander on a landing pad. OpenAI Gym provides a Lunar Lander environment that is designed to interface with reinforcement learning agents. Deep reinforcement learning is an exciting branch of AI that closely mimics the way human intelligence explores and learns in an environment. In this project, I used one kind of deep reinforcement learning called Deep Q-Learning (DQN).

# Ideas
This project is one of the assignments in the Machine Learning Specialization course on Coursera by DeepLearning.AI. I used some of the materials that I learned from the course. The most interesting thing is that the codes that previously ran well on the Coursera IDE have lots of bugs if run locally on the Jupyter Notebook (or any other local IDE). So, I am challenged to debug it even though I am new to this topic. Besides that, I am tuning the parameters and hyperparameters of the model and creating a video on how the model learns to land a rocket on a lunar lander. Note that this might be the best model, so any advice or recommendation would be much appreciated for my evaluation.

# Model and Hyperparameter
* Set the memory buffer to 100000.
* Set gamma to 0.99.
* Set the learning rate to 0.001.
* Set the maximum set to update to 4.
* Create a neural network with random weights for each state.
* Create a function to calculate the model's loss and learn agents.
* Loop the model to estimate the weight for each state until the mean of latest 100 epoch is greater than or equal than 200.
* Embed a video to see how the rocket learns to land.
* Visualize the reward.

# Videos
This is how the rocket learns to land. The full video is 2 hours and 46 minutes long. To summarize how a rocket learns, this video was made every 20 epochs.
https://user-images.githubusercontent.com/125811483/233581425-1e1cc132-463a-4d70-b979-bcc44c824814.mp4

The highest reward until the loop stop is 302.24. Reached after 476 retries.
https://user-images.githubusercontent.com/125811483/233581677-8c81833f-22bf-4b79-90ad-48d26dd6bc7f.mp4

# Reward
This is the reward for the overall epoch.
![image](https://user-images.githubusercontent.com/125811483/233580729-c6715d32-7c3c-41b7-9bae-32b5192a90d3.png)


