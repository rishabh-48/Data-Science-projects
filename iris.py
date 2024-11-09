import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("iris.data.csv")

# Encoding the 'Size' categorical variable
size_mapping = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
df['target'] = df['target'].map(size_mapping)

# Separating features and target variable
X = df[['sepal L', 'sepal W', 'petal L', 'petal W']]
y = df['target']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Create Logistic Regression classifier
log_reg = LogisticRegression()

# Fit the model
log_reg.fit(X_train, y_train)

# Calculate accuracy on the test set
accuracy = log_reg.score(X_test, y_test)

# Print the accuracy
print("Accuracy:", accuracy)