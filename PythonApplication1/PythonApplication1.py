
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Collect Data from Arduino

#features of x and y components
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#knn classifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

def plot_data_with_peaks(data, peaks):
    """
     Plots the pulse oximeter data and highlights the detected peaks.
    :param data: NumPy array of pulse oximeter data
    :param peaks: List of indices where peaks were detected
     """
    plt.plot(data, label='Pulse Oximeter Data')
    plt.plot(peaks, data[peaks], "x", label='Detected Peaks', color='red')
    plt.title("Pulse Oximeter Data from Arduino with Detected Peaks")
    plt.xlabel("Time (Sample Index)")
    plt.ylabel("Sensor Value")
    plt.legend()
    plt.show()
# Step 4: Estimate Heart Rate
