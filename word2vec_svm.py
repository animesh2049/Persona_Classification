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

# count = 0
# temp_labels = []
# labels_in_parts = []
# documents_in_parts = []
# temp_doc_embedding = []
# number_of_document_parts = (doc_id + 1) / document_parts
#
# for i in range(doc_id + 1) :
#     doc = document_embedings[i]
#     label = labels[i]
#     if count < number_of_document_parts :
#         temp_doc_embedding.append(doc)
#         temp_labels.append(label)
#         count += 1
#     else :
#         documents_in_parts.append(temp_doc_embedding)
#         labels_in_parts.append(temp_labels)
#         temp_doc_embedding = []
#         temp_labels = []
#         count = 0

# X_train, X_test, y_train, y_test = train_test_split(document_embedings, labels, test_size = test_size, random_state = 0)
# clf = svm.SVC()
# clf.fit(X_train, y_train)
# score = clf.score(X_test, y_test)
# print score

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