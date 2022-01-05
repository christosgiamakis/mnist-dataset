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

"""check dimension ofx_train, x_test, y_train, y_test."""

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

"""visualize input data"""

f, ax = plt.subplots(1, 10, figsize=(28,28))

for i in range (0,10):
  sample=x_train[y_train==i][0]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title('Label: {}'.format(i), fontsize=16)

"""visualize labels"""

for i in range(10):
  print(y_train[i])

"""normalizations in train and test sets"""

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

"""Τρέχουμε τον classifier KNN για n=1 και τυπώνουμε accuracy."""

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)

preds = model.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy)

"""Τρέχουμε τον classifier KNN για n=3 και τυπώνουμε accuracy."""

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

preds = model.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy)

"""Τρέχουμε τον classifier πλησιέστερου κέντρου και τυπώνουμε accuracy."""

model = NearestCentroid()
model.fit(x_train, y_train)

preds = model.predict(x_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy)

"""Στήνουμε το Νευρωνικό Δίκτυο."""

model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

sgd = SGD(lr=0.001)

model.compile(loss= 'mean_squared_error', optimizer=sgd, metrics=['accuracy'])

"""Θα εκπαιδεύσουμε το Νευρωνικό Δίκτυο."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Εκπαίδευση για άλλες 50 εποχές.

"""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Αξιολόγηση."""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Αλλαγή MSE σε CrossEntropy.

"""

model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

sgd = SGD(lr=0.001)

model.compile(loss= 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

"""Εκπαίδευση του Νευρωνικού δικτύου με τη νέα loss function."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Εκπαίδευση για άλλες 50 εποχές."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Αξιολόγηση.




"""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Εκπαιδεύουμε για άλλες 50 εποχές."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Εκπαίδευση για άλλες50 εποχές με μείωση του batch size σε 32."""

model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, verbose=2)

"""Αξιολόγηση."""

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Τροποποίηση Νευρωνικού με 32 νευρώνες στο κρυφό επίπεδο."""

model = Sequential()
model.add(Dense(32, input_shape=(784,), activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

sgd = SGD(lr=0.001)

model.compile(loss= 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

"""Εκπαίδευση Νευρωνικού με 32 νευρώνες στο κρυφό επίπεδο."""

model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, verbose=2)

"""Εκπαίδευση για άλλες 50 εποχές.

"""

model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, verbose=2)

"""Αξιολόγηση."""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Τροποποίηση του Νευρωνικού με αλλαγή του SGD σε adam optimizer."""

model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Εκπαίδευση Νευρωνικού."""

model.fit(x=x_train, y=y_train, batch_size=32, epochs=50, verbose=2)

"""Αξιολόγηση."""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))

"""Προσθήκη ενός ακόμα κρυφού στρωματος 64 νευρώνων."""

model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Εκπαίδευση του Νευρωνικού."""

model.fit(x=x_train, y=y_train, batch_size=64, epochs=50, verbose=2)

"""Αξιολόγηση."""

print('Train accuracy:', model.evaluate(x_train,y_train, batch_size=64))
print('Test accuracy:', model.evaluate(x_test,y_test, batch_size=64))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_pred_classes)

y_pred = model.predict(x_train)
print(y_pred[0], np.argmax(y_pred[0]))
