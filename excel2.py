import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
# Read the CSV file
df = pd.read_csv("Customer Purchasing Behaviors.csv")
print(df)


X = df[['annual_income']]
y = df['purchase_amount']

reg= LinearRegression()
m=len(X)
x=X.reshape((m,1))

reg=reg.fit(X,y)

y_pred= reg.predict(X)

rmse= np.sqrt(mean_squared_error(y,y_pred))
print(rmse)