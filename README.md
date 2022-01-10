# Unsupervised-and-Ensemble-machine-learning-algorithms
# Servo data set:
Servo dataset is dataset that forms from a simulation of servo system. It is a regression data set and it is consisting of a 167 instance and 5 features (167 * 5). Those features are consisting of 3 numerical features and 2 categorical features. So, I need label encoding to convert those categorical data into numerical to be handled. The target class to be predicted in this data set is the class feature for the system and this depends on the 4 features remaining which are the motor type, screw, pgain, and vgain. There are no missing value or outliers in this data. 
# Early-stage diabetes risk prediction dataset:
This dataset is a set in which record the signs and symptoms of the newly diagnosed diabetic or the patient that would be diabetic diagnosed. This is a classification dataset which classify if a one has a diabetic or not. So, our target class is class attribute which the attribute to be predicted. This dataset contains of 520 instance and 17 features (520 * 17). These features are age, gender, weakness etc..., all of these features are categorial data except for the age so I also need to apply a label encoder to it.
# Morphological dataset: 
Morphological dataset is a dataset of handwritten numerals (0 - 9). It consists of 200 instances per class (for a total of 2,000 instances) have been digitized in binary images. In this dataset, these digits are represented in terms of 6 morphological features. The features type is integer, so it does not need encoding or any kind of transformation. But still, it’s needed a scaling and PCA (Principal Component Analysis) because when the data plotted it shows a high variance between features. The dataset doesn’t contain any kind of missing values (NaN). The data doesn’t contain outliers. The dimensions need to be reduced by using PCA. The target class in this dataset is (‘Class’) it’s contained the number of classes from (1 to 10).  

# Clustering algorithm. 

I used Kmeans algorithm for clustering the early-stage dataset and I give the algorithm 2 clusters to form based on the elbow algorithm that is applied to our data set. This elbow algorithm helps us in determining the number of clusters I should give to our clustering algorithm.
# Applying SVC and decision tree on Morphological dataset: 

I used the SVC and Decision tree models as an individual model then I made an ensemble model with them and adding the voter classifier in the two methods the soft method and the hard method. I found that the (‘Soft’) method is giving a higher accuracy the reason behind that is the Soft method give more weight to highly confident votes.

![1](https://user-images.githubusercontent.com/96385070/148804527-0262111d-3eb5-4b0f-bb77-d838078d0e8f.JPG)

# HARD VOTE

![2](https://user-images.githubusercontent.com/96385070/148804757-e3bcdb6a-d34b-4075-a4ac-0854a7d243c6.JPG)

# SOFT VOTE

![3](https://user-images.githubusercontent.com/96385070/148805111-e3ee76bc-09d9-47ee-ad99-1769a0844de9.JPG)

# Applying logistic regression on Early-stage diabetes risk prediction dataset
I first loaded our dataset and apply a label encoder to convert the categorical data into numerical data and then I normalized the data for the efficiency in training. And after that I create our logistic regression model and fit it to the data. And after training the accuracy was about 90%.So, I start to implement a PCA algorithm for dimension or feature reduction and the data has become a data of dimension 520 * 2 instead of 520 * 17. And then after feature reduction I applied the logistic regression mode again using the new extracted data and it give us an accuracy of 95 %. and this proves a clearly improvements in the accuracy after using the PCA algorithm as a feature extractor. 

![4](https://user-images.githubusercontent.com/96385070/148806720-187ec350-654c-4768-9b77-b3a242e77ebe.JPG)

Then I applied an ensemble model which consists of 3 models and those models are svm and logistic regression and random forest. And the obtained accuracy of this ensemble model was 96 % which is much better than the logistic algorithm I have applied. And this is a comparison between the 3 models that is used in the ensemble model as individual and between the ensemble model accuracy. 

![5](https://user-images.githubusercontent.com/96385070/148806890-c70715e4-df71-4764-b85f-28561c66b4a1.JPG)

So, I obtained that the voting classifier and the svm and random forest gives us the same accuracy percentage which is better than the logistic regression algorithm. 
# Applying a linear regression model to the servo dataset.
I first loaded our regression data set and convert it into csv file. And then I make a check for the outliers and the null values and I found nothing. And after that I applied a label encoder to those data to convert the categorical data into numerical data. And then I normalized the data I have and after that I created the linear regression model and fit the model to the train data. And then I print out the r2 score and it was about 68% which is a poor accuracy. So, I applied a PCA algorithm to improve the accuracy of the model by plotting the principal components variance and I then chased the first 4 PC in as they have high variance and I dropped the last PC. And then I plot the data and it seem like I have 4 clusters in our data. And then I apply the linear regression model again after feature reduction and the accuracy now is much better and it was about 98 %.

![6](https://user-images.githubusercontent.com/96385070/148807257-1cae12f7-c9ab-48c0-80d2-b7dce0659877.JPG)

I can say that applying the PCA algorithm was very efficient and it really improves our accuracy. 

# Bias and Variance: 

Bias is the difference between the Predicted Value and the Expected Value. Variance is the amount that the estimate of the target function will change if different training data was used. A model with a high bias error underfits data and makes very simplistic assumptions on it and the model with a high variance error overfits the data and learns too much from it and the good model is where both Bias and Variance errors are balanced. In the decision tree model the bias was 1.7 and the variance was 1.1 so that means that the bias and the variance are balanced, and the model doesn’t underfit or over fit. 

# Imbalanced Data:

An imbalanced data is a data with an unequal class. Example is the target class consist of positive and negative. So, when the negative is majority and the positive is the minority then that an imbalanced data. Here in the dataset of diabetes the negative is 320 and the positive is 200. So that makes the data imbalanced. First, I need to transform the strings (positive and negative in to 0 - 1) using encoding method. The handling of this type of data divided into three methods. The first is the over sampling and the second is the under sampling and the third is the SMOTE. I tried the three models and the ROC curve and confusion matrix. Here I will see the ROC curve.

![download (4)](https://user-images.githubusercontent.com/96385070/148807576-dcbd7f2a-5a2f-4cd8-9448-31f094663605.png)

When I compare the three methods to each other (Under sampling, Over Sampling and SMOTE) I define the table below: 

![Capture](https://user-images.githubusercontent.com/96385070/148807686-446d2a29-4e58-4831-b24e-1136981203c7.JPG)
All these methods used the decision tree classifier.  
