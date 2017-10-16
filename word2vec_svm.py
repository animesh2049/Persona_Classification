import csv
import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

document_embedings = []
with open('document_embeddings_from_skipthoughts.csv', 'r') as document_file:
    reader = csv.reader(document_file)
    for row in reader:
        document_embedings.append(row)

label_file = open('labels.pickle', 'r')
vector_dimension = 4800
labels = pickle.load(label_file)
count = 0
labels = labels[:300]
test_size = 0.1

X_train, X_test, y_train, y_test = train_test_split(document_embedings, labels, test_size = test_size, random_state = 0)
clf = svm.SVC()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print score

# document_embedings = np.array(document_embedings)
# labels = np.array(labels)
# model = Sequential()
# model.add(Dense(100, input_dim=vector_dimension, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(document_embedings, labels, epochs=100, batch_size=10, validation_split=test_size)
# scores = model.evaluate(document_embedings, labels)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
