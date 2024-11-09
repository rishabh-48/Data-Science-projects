import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Read the CSV file
df = pd.read_csv("diabetes.csv")
print(df)

# # Convert categorical variables into numerical variables
# le = LabelEncoder()
# df['Gender'] = le.fit_transform(df['Gender'])

# Assume that the Excel file has multiple columns: 'Gender', 'Age', 'EstimatedSalary', and 'Purchased'
X = df[['Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness',	'Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age']]
y = df['Outcome']
print(df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train)
print(X_test)

# Create a logistic regression model
model = LogisticRegression()  
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Print the accuracy
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

# Print the classification report
print('Classification Report:')
print(metrics.classification_report(y_test, y_pred))

# Print the confusion matrix
print('Confusion Matrix:')

print(metrics.confusion_matrix(y_test, y_pred))

# Create a new data point with Gender 1, Age 30, and EstimatedSalary 50000
new_data = pd.DataFrame({'Pregnancies': [3], 'Glucose': [78], 'BloodPressure': [50], 'SkinThickness': [32] ,'Insulin': [88], 'BMI': [31] ,'DiabetesPedigreeFunction': [0.248], 'Age': [26],})

# Make a prediction using the model
predicted_outcome = model.predict(new_data)

print('Predicted outcome:', predicted_outcome[0])

import seaborn as sns

# Create confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Outcome', 'Outcome'], 
            yticklabels=['Not Outcome', 'Outcome'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()