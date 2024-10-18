
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#load dataset
iris = pd.read_csv('Iris.csv')

#features of x and y components
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#knn classifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Predicted labels: ", y_pred)
print("True labels: ", y_test)
print(f"Accuracy: {accuracy * 100:.2f}%", )
