import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Prepare the data
weights = np.array([51, 62, 69, 64, 65, 56, 58, 57, 55])
heights = np.array([167, 182, 176, 173, 172, 174, 169, 173, 170])
classes = np.array(['underweight', 'normal', 'normal', 'normal', 'normal', 'underweight', 'normal', 'normal', 'normal'])

# Combine weights and heights into a single feature array
X = np.column_stack((weights, heights))
y = classes

# Step 2: Create the KNN model
knn = KNeighborsClassifier(n_neighbors=3)  # You can choose the number of neighbors

# Step 3: Fit the model to the data
knn.fit(X, y)

# Step 4: Make a prediction for the new sample
new_sample = np.array([[57, 170]])
predicted_class = knn.predict(new_sample)

print(f'The predicted class for weight 57 kg and height 170 cm is: {predicted_class[0]}')