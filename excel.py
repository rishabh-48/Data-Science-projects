import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

try:
    # Read the CSV file
    df = pd.read_csv("vehicles.csv")
    print(df)
except FileNotFoundError:
    print("The file does not exist at the specified location.")
except PermissionError:
    print("The script does not have permission to read the file.")
except Exception as e:
    print("An error occurred: ", str(e))

if 'df' in locals():
    try:
        # Assume that the Excel file has two columns: 'head size' and 'brain weight'
        X = df[['annual_income']]
        y = df['purchase_amount']
    except KeyError:
        print("The columns 'annual_income' and 'purchase_amount' are not found in the Excel file.")
    else:
        try:
            # Check if the data types of the columns are numeric
            if not pd.api.types.is_numeric_dtype(X['annual_income']) or not pd.api.types.is_numeric_dtype(y):
                print("The data types of the columns 'annual_income' and 'purchase_amount' must be numeric.")
            else:
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                # Create a linear regression model
                model = LinearRegression()  
                model.fit(X_train, y_train)

                # Make predictions using the testing set
                y_pred = model.predict(X_test)

                # Print the coefficients
                print('Coefficient of Determination (R^2):', model.score(X_test, y_test))
                print('Intercept:', model.intercept_)
                print('Slope:', model.coef_)

                # Create a new data point with annual income 25000
                new_data = pd.DataFrame({'annual_income': [25000]})

                # Make a prediction using the model
                predicted_purchase_amount = model.predict(new_data)

                print('Predicted purchase amount:', predicted_purchase_amount[0])

                # Create a scatter plot of the data
                plt.figure(figsize=(10,6))
                sns.scatterplot(x='annual_income', y='purchase_amount', data=df)

                # Plot the regression line
                plt.plot(X_test, y_pred, color='red')

                # Set labels and title
                plt.xlabel('annual_income')
                plt.ylabel('purchase_amount')
                plt.title('Linear Regression of annual_income vs purchase_amount')

                # Show the plot
                plt.show()

                
        except Exception as e:
            print("An error occurred: ", str(e))