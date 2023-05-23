# Importing necessary libraries for data analysis and visualization

# Importing pandas library for data manipulation and analysis
import pandas as pd
# Importing numpy library for scientific computing
import numpy as np
# Importing seaborn library for creating informative and attractive statistical graphics
import seaborn as sns
# Importing matplotlib.pyplot library for creating static, animated, and interactive visualizations in Python
import matplotlib.pyplot as plt



# Reading the data file using pandas library and storing its contents in a DataFrame object
df=pd.read_csv("C:/Users/91952/Documents/framingham_heart_disease_data.csv")


# Displaying the column names of the DataFrame
df.columns

# Counting the number of missing values in each column of the DataFrame
df.isnull().sum()

# Displaying the first few rows of the DataFrame
df.head()


#since education has vey less influence in this prediction and it having a large number of null values, drop the attribute

data = df.drop(['education'], axis = 1)
data.head()


### Finding mean of attributes and replacing null values with mean values

# Calculating the mean values of certain columns of the DataFrame

mean_cigsPerDay = round(data["cigsPerDay"].mean())
mean_BPmeds = round(data["BPMeds"].mean())
mean_totChol = round(data["totChol"].mean())
mean_BMI = round(data["BMI"].mean())
mean_glucose = round(data["glucose"].mean())
mean_heartRate = round(data["heartRate"].mean())


# Replace missing values with Mean values

data['cigsPerDay'].fillna(mean_cigsPerDay, inplace = True)
data['BPMeds'].fillna(mean_BPmeds, inplace = True)
data['totChol'].fillna(mean_totChol, inplace = True)
data['BMI'].fillna(mean_BMI, inplace = True)
data['glucose'].fillna(mean_glucose, inplace = True)
data['heartRate'].fillna(mean_heartRate, inplace = True)


# Ensure that all missing values are filled

data.isnull().sum()


# ### Analysis through visualizing data


# Creating a pairwise scatter plot of selected columns in the DataFrame

sns.pairplot(data[["age","cigsPerDay","totChol","sysBP","diaBP","BMI","heartRate","glucose"]]);


# Splitting the DataFrame into predictor variables and target variable

X = data.drop('TenYearCHD',axis=1)

# Load the target variable to y

y=data['TenYearCHD']

# Train/Test splitting of data 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)

# Initializing Logistic Regression Model

from sklearn.linear_model import LogisticRegression
Classifier = LogisticRegression()

# Train the Logistic Regression Model

Classifier.fit(X_train,y_train)


# Predicting the target variable using the trained model and the test data
y_test_hat = Classifier.predict(X_test)

# Creating a DataFrame to store the actual and predicted values of the target variable
Results = pd.DataFrame({'Actual': y_test, 'Predictions': y_test_hat})

# Displaying the first 5 rows of the Results DataFrame
Results.head(5)


# Importing the accuracy_score function from the scikit-learn library
from sklearn.metrics import accuracy_score

# Calculating the accuracy score of the predicted values
print(accuracy_score(y_test,y_test_hat))

# Predicting the target variable for the training data
y_train_hat = Classifier.predict(X_train)



# Calculating the accuracy score of the predicted values for the training data
print(accuracy_score(y_train, y_train_hat))


y_test_hat_proba = Classifier.predict_proba(X_test)
y_test_hat_proba


# Import the confusion_matrix function from sklearn.metrics module
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix for the test set and output it to the console.
cm = confusion_matrix(y_test,y_test_hat)
print(cm)



# Importing the seaborn library for visualization
import seaborn as sn

# Creating a heatmap using the confusion matrix with annotations
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)

# Setting the title, x-label and y-label of the heatmap
plt.title('Confusion Matrix - Test Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')



# Displaying an image of the confusion matrix with the Wistia color map
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

# Creating a list of class names and setting the title, x-label and y-label of the plot
classNames = ['Heart Disease', 'NO Heart Disease']
plt.title('Confusion Matrix-Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Setting the ticks and labels for the x and y axes
tick_marks = np.arange(2)
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)

# Adding the values of True Negatives, False Positives, False Negatives and True Positives to the plot
s = [['TN','FP'],['FN','TP']]
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j])+"= "+str(cm[i][j]))

# Displaying the plot
plt.show()


# Import the classification_report function from sklearn.metrics module
from sklearn.metrics import classification_report

# Call the classification_report function with y_test and y_test_hat as arguments and print the report, which includes metrics such as precision, recall, F1-score, and support for each class in the classification model

print(classification_report(y_test,y_test_hat))



# Calculate metrics values individually
# Assigning Variables for convinience

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]




# Calculating precision, recall, specificity, and accuracy mathematically by using their equations 

recall = TP / (TP + FN )
print("Recall= ",recall)


precision = TP / (TP + FP)
print("Precision=",precision)


specificity = TN /  (TN + FP)
print("Specificity = ", specificity)


accuracy = ( TP + TN ) / ( TP + TN + FP + FN)
print("Accuracy =" , accuracy)
