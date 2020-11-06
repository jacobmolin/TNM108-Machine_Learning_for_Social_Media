# Answers to Lab3

## Part 1: Explain the concepts and Gaussian NB algorithm

The concept is a classifier based on the Bayes Theorem with the assumption of strong independence between features and feauture-value independence (Naive). Gaussian Naive Bayes works with continous data where it is assumed that the continous values in each class are distributed to Gaussian (normal)distribution.

The GNB classifier calculates for all classes the distance between the data point and the class mean, divided by the class' standard deviation.

A simple model can be created by assuming that the data is described by a Gaussian (normal) distrubution with no covariance.

***********************************************************

## Part 2

1. Explain what the program does.
    The program uses different mathematical methods to complete the lower half of an image representing a face.

2. What is your interpretation of the final plot? Which algorithm has better performance in building the unknown parts of the face?
    Extra trees: gives a somewhat recognizable lower part of the face, although it's pretty dizzy/blurry/unfocused
    K-NN: Similar to ET, but a bit more focused and 'shakey'
    Linear regr: gives a lower part that looks crazy. The intensity values does not match the original image at all.
    Ridge: Also similair to ET and K-NN but blurrier. The contours are barely visible

    The best algorithm that has the better performance is extra trees or K-NN algorithm

3. Download the code from the link above and modify it by adding the results of the following
algorithms to the final plot:
(a) Regression decision tree with max depth of 10 and max number of features 50
(b) Regression decision tree with max depth of 20 and max number of features 50
(c) Regression decision tree with max depth of 20 and max number of features 25
(d) Random forest with max depth of 10 and max number of features 50
(e) Random forest with max depth of 20 and max number of features 50
(f) Random forest with max depth of 20 and max number of features 25
How do you interpret the results?

RDT:
depth gives...
More features gives...

RGR: 
depth gives...
More features gives...
More estimators gives better result because more trees are used to calculate the average.

4. How could performance of random forest be improved?
To use more estimators, more trees to calculate a more accurate mean
***********************************************************

Part 3:

1. In the script of the Regression section (the one before the section RFE), we apply cross
validation with 10 folds. Note that the script does not make any change to the dataset.
Modify the script in oder to reshuffle the rows of the data set to randomize the cross
validation folds before applying the cross validation.
Run again the script but on the reshuffled data set and re-calculate the MSE and R2
scores. Do you obtain a better performance?
To know more about reshuffling You can read the part about reshuffling in Section 3.1
”Cross-validation: evaluating estimator performance” of scikit-learn1

2. What happens if you do reshuffling and RFE? do you get better results than only reshuffling?

3. In the section Car Evaluation Quality, we performed the evaluation metrics for linear
support vector machine, naive bayes, logistic regression and k nearest neighbours. As you
can see at page 18, they do have a poor performance.
Find out if there are ML algorithms that perform better on the data cars.csv data set.
You may test decision trees and random forest as well as other type of SVM.
