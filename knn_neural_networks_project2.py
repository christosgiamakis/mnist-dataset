# -*- coding: utf-8 -*-
"""KNN Neural Networks project2.ipynb



Load the libraries that we will use
"""

# Commented out IPython magic to ensure Python compatibility.
#Libraries for mid-project
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score

#Libraries for final project
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# %matplotlib inline
np.random.seed(42)

"""Load Dataset"""

from keras.datasets import mnist
(x_train, y_train), (x_test,y_test)=mnist.load_data()

"""Check dimension ofx_train, x_test, y_train, y_test."""

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

"""Visualize input data"""

f, ax = plt.subplots(1, 10, figsize=(28,28))

for i in range (0,10):
  sample=x_train[y_train==i][0]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title('Label: {}'.format(i), fontsize=16)

"""Visualize labels"""

for i in range(10):
  print(y_train[i])

"""Normalizations in train and test sets"""

#Convert x_train, x_test to vectors
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
print(x_train.shape)
print(x_test.shape)

#normalize input data
x_train = x_train / 255.0
x_test = x_test / 255.

#One hot encoding to labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
for i in range(10):
  print(y_train[i])

"""Run KNN classifier for n=1 and print accuracy."""

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)

preds = model.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy)

"""Run KNN classifier for n=3 and print accuracy."""

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

preds = model.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy)

"""Run NearestCentroid classifier and print accuracy."""

model = NearestCentroid()
model.fit(x_train, y_train)

preds = model.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy)

"""Neural Network."""

model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

sgd = SGD(lr=0.001)

model.compile(loss= 'mean_squared_error', optimizer=sgd, metrics=['accuracy'])

"""Train Neural Network."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Train for 50 epochs."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Evaluation."""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Change lossfunction from MSE to Crossentropy."""

model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

sgd = SGD(lr=0.001)

model.compile(loss= 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

"""Train with new loss function."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Train for 50 epochs."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Evaluation."""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Train for 50 epochs."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Train for 50 epochs and reduce batch size to 32"""

model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, verbose=2)

"""Evaluation."""

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Change NN by putting 32 neurons in hidden layer."""

model = Sequential()
model.add(Dense(32, input_shape=(784,), activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

sgd = SGD(lr=0.001)

model.compile(loss= 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

"""Train model."""

model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, verbose=2)

"""50 epochs more."""

model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, verbose=2)

"""Evaluation."""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Adam optimizer."""

model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Train."""

model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, verbose=2)

"""Evaluation."""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Add another hidden layer with 64 neurons."""

model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Train."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Evaluation."""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))
