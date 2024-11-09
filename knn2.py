import pandas as pd

df=pd.read_csv("iris.data.csv")


# Encoding the 'Size' categorical variable
size_mapping = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
df['target'] = df['target'].map(size_mapping)

from sklearn.preprocessing import StandardScaler

# Separating features and target variable
X = df[['sepal L',	'sepal W',	'petal L',	'petal W']]
y = df['target']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# # Fit the model
# knn.fit(X_train, y_train)

# import numpy as np

# # Input features
# input_features = np.array([[140, 7, 2]])  # 2 corresponds to 'Medium'

# # Standardize the input features using the same scaler
# input_scaled = scaler.transform(input_features)

# # Make the prediction
# predicted_fruit_type = knn.predict(input_scaled)

# print("Predicted Fruit Type:", predicted_fruit_type[0])

# Fit the model
knn.fit(X_train, y_train)

# Calculate accuracy on the test set
accuracy = knn.score(X_test, y_test)

# Print the accuracy
print("Accuracy:", accuracy)