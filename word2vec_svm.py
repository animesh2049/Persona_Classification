import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

document_file = open('document_embeddings', 'r')
label_file = open('labels', 'r')
vector_dimension = 300

document_embedings = pickle.load(document_file)
labels = pickle.load(label_file)
test_size = 0.1

document_embedings = np.array(document_embedings)
labels = np.array(labels)
model = Sequential()
model.add(Dense(100, input_dim=vector_dimension, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(document_embedings, labels, epochs=100, batch_size=10, validation_split=test_size)
scores = model.evaluate(document_embedings, labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))