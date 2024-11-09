import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('Craigslist Car Dataset.csv')
 # Handle missing values
data.dropna(inplace=True)  # or use data.fillna(method='ffill')

# Define features and target variable
X = data[['year', 'make', 'condition', 'odometer']]
y = data['price']
print(X)
print(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import RobustScaler

import numpy as np

# Apply logarithmic transformation to 'odometer' feature
X_train['odometer'] = np.log(X_train['odometer'])
X_test['odometer'] = np.log(X_test['odometer'])

# Preprocessing: One-hot encode categorical variables and scale numerical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['year', 'odometer']),  # Include 'odometer' in the 'num' transformer
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['make', 'condition'])  # Add handle_unknown='ignore'
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from the coefficients
feature_importance = np.abs(model.coef_)
features = preprocessor.get_feature_names_out()

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance in Predicting Car Resale Price')
plt.show()

# Correlation heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

from sklearn.feature_selection import SelectKBest, f_regression

# Select the top features
selector = SelectKBest(score_func=f_regression, k='all')
X_new = selector.fit_transform(X_train, y_train)

# Train a new model with selected features
model_optimized = LinearRegression()
model_optimized.fit(X_new, y_train)

# Evaluate the optimized model
y_pred_optimized = model_optimized.predict(selector.transform(X_test))
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

print(f'Optimized Mean Squared Error: {mse_optimized}')
print(f'Optimized R-squared: {r2_optimized}')

# #bonus challenge
# from sklearn.linear_model import Ridge, Lasso

# # Initialize models
# ridge_model = Ridge(alpha=1.0)
# lasso_model = Lasso(alpha=0.1)

# # Train and evaluate Ridge
# ridge_model.fit(X_train, y_train)