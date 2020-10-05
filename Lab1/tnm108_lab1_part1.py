import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

print("***** Train_Set *****")
print(train.head())
print("\n")
print("***** Test_Set *****")
print(test.head())
print("***** Train_Set Describe *****")
print(train.describe())
print("\n")
print("***** Test_Set Describe *****")
print(test.describe())

print(train.columns.values)
print("\n")

# values missing in data, lets see where they are:
# For the train set
train.isna().head()
# For the test set
test.isna().head()

#Let's get the total number of missing values in both datasets.
print("*****Missing in the train set*****")
print(train.isna().sum())
print("\n")
print("*****Missing in the test set*****")
print(test.isna().sum())
print("\n")


#Pandas provides the fillna() function for replacing missing values with a specific value.
# Let's apply that with Mean Imputation.
# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(), inplace=True)

# time to see if the dataset still has any missing values.
print("*****Missing in the train set after imputation*****")
print(train.isna().sum())
print("\n")

# Let's see if you have any missing values in the test set.
print("*****Missing in the test set after imputation*****")
print(test.isna().sum())

# Categorical: Survived, Sex, and Embarked. Ordinal: Pclass. (Non-numeric?)
# Continuous: Age, Fare. Discrete: SibSp, Parch. (Numerical?)


#Ticket is a mix of numeric and alphanumeric data types.
train['Ticket'].head()

# Cabin is alphanumeric
train['Cabin'].head()

#Survival count with respect to Pclass:
train[['Pclass', 'Survived']].groupby(['Pclass'],
as_index=False).mean().sort_values(by='Survived', ascending=False)

# Survival count with respect to Sex:
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',
ascending=False)

#Survival count with respect to SibSp:
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
ascending=False)

# Let's first plot the graph of "Age vs. Survived":
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# plt.show()


train.info()

train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])
# Let's investigate if you have non-numeric data left

train.info()

test.info()

X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])

train.info()

kmeans = kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or not survived.
kmeans.fit(X)

# KMeans(algorithm='elkan', copy_x=True, init='k-means++', max_iter=30000, n_clusters=2, n_init=10,
# n_jobs=100, precompute_distances='auto', random_state=None, tol=0.0001, verbose=0)


# looking at the percentage of passenger records that were clustered correctly.
correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1 


print("\nValidation score: ", correct/len(X))

kmeans = kmeans = KMeans(n_clusters=2, max_iter=6000, algorithm='elkan')
kmeans.fit(X)
# KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600, n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001, verbose=0)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("Validation score: ", correct/len(X))

# kmeans = kmeans = KMeans(n_jobs=2, max_iter=6000, algorithm='elkan')
# kmeans.fit(X)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
# KMeans(algorithm='full', copy_x=True, init='k-means++', max_iter=600000,
#  n_clusters=2, n_init=10, n_jobs=10000, precompute_distances='auto',
#  random_state=None, tol=0.0001, verbose=0)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1
        
print("Edited validation score: ", correct/len(X))

# 3. 