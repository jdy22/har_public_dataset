import numpy as np
import pickle

with open("data_test2.pk1", "rb") as file:
    data = pickle.load(file)
x_train = data[0]
x_test = data[1]
y_train = data[2]
y_test = data[3]

train_labels = np.argmax(y_train, axis=1)
test_labels = np.argmax(y_test, axis=1)

train_labels, train_counts = np.unique(train_labels, return_counts=True)
test_labels, test_counts = np.unique(test_labels, return_counts=True)

print(train_labels)
print(train_counts)
print(test_labels)
print(test_counts)