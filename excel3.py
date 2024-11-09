import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split #sklearn is used for implementing machine learning model suggets lienear rgression,logistic regression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# try:
    # Read the CSV file
df = pd.read_csv("Social_Network_Ads.csv")
print(df)
# except FileNotFoundError:
#     print("The file does not exist at the specified location.")
# except PermissionError:
#     print("The script does not have permission to read the file.")
# except Exception as e:
#     print("An error occurred: ", str(e))

# # if 'df' in locals():
# #     try:
# #         # Convert categorical variables into numerical variables
# #         le = LabelEncoder()
# #         df['region'] = le.fit_transform(df['region'])

#         # Assume that the Excel file has multiple columns: 'age', 'annual_income', 'loyalty_score', 'region', 'purchase_frequency', and 'purchase_amount'
# X = df[['Gender', 'Age', 'EstimatedSalary']]
# y = df['Purchased']
#     # except KeyError:
#     #     print("The columns 'Gender', 'Age', 'EstimatedSalary', 'purchase_frequency', and 'purchase_amount' are not found in the Excel file.")
#     # else:
#     #     try:
#     #         # Check if the data types of the columns are numeric
#     #         if not pd.api.types.is_numeric_dtype(X['age']) or not pd.api.types.is_numeric_dtype(X['annual_income']) or not pd.api.types.is_numeric_dtype(X['loyalty_score']) or not pd.api.types.is_numeric_dtype(X['region']) or not pd.api.types.is_numeric_dtype(X['purchase_frequency']) or not pd.api.types.is_numeric_dtype(y):
#     #             print("The data types of the columns 'age', 'annual_income', 'loyalty_score', 'region', 'purchase_frequency', and 'purchase_amount' must be numeric.")
#     #         else:
#     #             # Split the data into training and testing sets
#     #             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#     #             # Create a multiple linear regression model
#     #             model = LinearRegression()  
#     #             model.fit(X_train, y_train)

#     #             # Make predictions using the testing set
#     #             y_pred = model.predict(X_test)

#     #             # Print the coefficients
#     #             print('Coefficient of Determination (R^2):', model.score(X_test, y_test))
#     #             print('Intercept:', model.intercept_)
#     #             print('Coefficients:', model.coef_)

#     #             # Create a new data point with age 30, annual income 50000, loyalty score 5, region 1, and purchase frequency 10
#     #             new_data = pd.DataFrame({'age': [30], 'annual_income': [50000], 'loyalty_score': [5], 'region': [1], 'purchase_frequency': [10]})

#     #             # Make a prediction using the model
#     #             predicted_purchase_amount = model.predict(new_data)

#     #             print('Predicted purchase amount:', predicted_purchase_amount[0])

#     #             # Create a 3D scatter plot of the data
#     #             fig = plt.figure(figsize=(10,6))
#     #             ax = fig.add_subplot(111, projection='3d')
#     #             ax.scatter(X_train['age'], X_train['annual_income'], y_train)

#     #             # Set labels and title
#     #             ax.set_xlabel('age')
#     #             ax.set_ylabel('annual_income')
#     #             ax.set_zlabel('purchase_amount')
#     #             ax.set_title('Multiple Linear Regression of age, annual_income vs purchase_amount')

#     #             # Show the plot
#     #             plt.show()

#     #     except Exception as e:
#     #         print("An error occurred: ", str(e))