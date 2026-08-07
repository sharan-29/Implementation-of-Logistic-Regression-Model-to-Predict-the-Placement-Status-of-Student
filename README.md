# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the placement dataset, remove unnecessary columns, check for missing/duplicate values, and encode categorical features using Label Encoding.


2.Split the processed data into input features (X) and target variable (status), then divide it into training and testing datasets.


3.Train a Logistic Regression model using the training data and predict placement status for the test data as well as for a new student.


4.Evaluate the model using Accuracy and Classification Report (Precision, Recall, and F1-score) to measure prediction performance.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHARAN KUMAR G
RegisterNumber:  212224230260
*/
# 1. Import Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Placement_Data.csv")

# View first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

data1 = data.copy()

# Dropping 'sl_no' (serial number) and 'salary' (not needed for predicting placement)
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())

print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())

# 6. Define Features (X) and Target (y)
# X = all columns except 'status'
X = data1.iloc[:, :-1]
# y = 'status' column
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

lr = LogisticRegression(solver="liblinear")

# Train the model
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)

# 10. Evaluate Model Performance
# Accuracy: percentage of correctly predicted labels
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Classification Report: precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])


```

## Output:

<img width="706" height="517" alt="image" src="https://github.com/user-attachments/assets/1d2332c8-5bad-4dd5-ade4-d6ac8bbc3646" />

<img width="697" height="602" alt="image" src="https://github.com/user-attachments/assets/a9b927be-04fa-4bfe-8ecc-1cebf1e5a2b8" />


<img width="762" height="862" alt="image" src="https://github.com/user-attachments/assets/613887ee-52e5-490b-bfdc-a1fcc01fde16" />






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
