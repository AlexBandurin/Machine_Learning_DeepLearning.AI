# Machine_Learning Specialization Assignments

Here are my programming assignments from the Machine Learning Specialization by DeepLearning.AI, taught by Andrew Ng.
I got to explore the fundamental concepts in ML. 

This specialization contains 3 courses:

- Course 1: Supervised Machine Learning: Regression and Classification
- Course 2: Advanced Learning Algorithms
- Course 3: Unsupervised Learning, Recommenders, Reinforcement Learning

Below are links to the some of the notebooks in this repository along with the topics covered

- **Gradient descent** 
  [Gradient Descent - Course 1](https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/C1_W1_Lab04_Gradient_Descent_Soln.ipynb)
 
- **Linear Regression**    
  [Linear Regression - Course 1](https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/C1_W2_Linear_Regression.ipynb)
  
- **Logistic Regression**      
  [Logistic Regression - Course 1](https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/C1_W3_Logistic_Regression.ipynb)
  
- **Bias and Variance**     
  [Diagnosing Bias and Variance - Course 2](https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/C2W3_Lab_02_Diagnosing_Bias_and_Variance.ipynb)
  
- **Decision Trees**     
  [Decision Trees - Course 2](https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/C2_W4_Lab_01_Decision_Trees.ipynb)
  
- **K-Means Clustering**     
  [K-means - Course 3](https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/C3_W1_KMeans_Assignment.ipynb)
  
- **Anomaly Detection**     
  [Anomaly Detection - Course 3](https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/C3_W1_KMeans_Assignment.ipynb)
  

- **Neural Networks** (for handwritten digit recognition. Multiclass Classification.)<br>
  [Neural_Networks.ipynb - Course 2](https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/Neural_Networks.ipynb)
  
<font size=”10”>Digit Recognition Project overview:</font>

Here, I trained a neural network to recognize handwritten digits 0-9 using **Tensorflow** and **Keras**.
The model features 2 hidden layers with a ReLU activation functions and an output layer with a Softmax function 
to which a linear activation function is applied. Thus, each output is categorized into the appropriate "digit" category. 
Below is a diagram:
<p align="center">
<img src="https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/Digit_Recognition_model.jpeg"  width="60%" height="60%">
</p> 
<font size=”5”>Methodology:</font>

The model is trained on 5000 training images of handwritten digits in black-and-white such that each image consists of 20x20 pixels, 
and each pixel is represented by a floating-point number that indicates its grayscale intensity. Each image is recorded in a vector 
(of length 400), and each vector becomes a row in the training matrix (5000 x 400). The model was trained in 100 epochs. <br> The model 
outputs a vector of length 10, which contains probability values for each digit. The assigned label for each given image corresponds to 
the digit with the largest probability. <br>

<font size=”8”>Results:</font>
64 randomly-selected digits from the training set:
<p align="center">
<img src="https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/Digit_Recognition_training.jpeg"  width="60%" height="60%">
</p> 
We tested the model on 5000 images, and here are the results:
<p align="center">
<img src="https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/Digit_Recognition_Results.jpeg"  width="60%" height="60%">
</p> 
The accuracy was fairly high, with only 15 miscrassified images. These are shown below: 
<p align="center">
<img src="https://github.com/AlexBandurin/Machine_Learning_DeepLearning.AI/blob/master/Digit_Recognition_Errors.png"  width="60%" height="60%">
</p> 









