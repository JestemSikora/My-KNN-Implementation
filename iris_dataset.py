from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from Plotting_data import PlottingData
from KNN_from_scratch import KNN_Algorithm
import numpy as np

iris = datasets.load_iris()
k = 7
# Dataset
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

# Visualizing Data
figure, axis = plt.subplots(2, 2, figsize=(7,4))
graph = PlottingData(X_train, y_train, iris, axis)
figure.tight_layout()

# KNN Model - from scratch
clf_scratch = KNN_Algorithm(X_train, y_train, k)
clf_scratch.fit(X_train, y_train)

# Predecting 
idx_ = np.random.choice(X_test.shape[0])
pred = clf_scratch.prediction(X_test[idx_])

print(f'Our chosen X: {X_test[idx_]}')
print(f'Prediction: {pred}')
print(f'Real class: {y_test[idx_]}')

graph.Plotting_Function(X_test[idx_], pred, iris, axis)

# Testing Accuracy on KNN from scratch
correct_array = []
for i in range(len(X_test)):
    idx = np.random.choice(X_test.shape[0], replace=False)
    pred = clf_scratch.prediction(X_test[idx])
    if pred == y_test[idx]:
        correct_array.append(pred)

acc = len(correct_array)/len(y_test)
print(f"Accuracy: {acc:.2f}")



# Scikit-Learn
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train,y_train)
X_test_pred = X_test.reshape(1,-1)
# print(neigh.predict(X_test_pred[idx_]))

plt.show()