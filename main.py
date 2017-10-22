import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import os
import sys
import downsample

WORD_EMBEDDING_FILE_NAME = "word_embeddings.npy"  # For other cases this can also be taken as input
SENTENCE_EMBEDDING_FILE_NAME = "sentence_embeddings_from_skip_thoughts.npy"
LABEL_FILE_NAME = "labels.npy"
vector_dimension = (1, 2400)
labels = np.load(LABEL_FILE_NAME)
# word_embeddings = np.load(WORD_EMBEDDING_FILE_NAME)
test_size = 0.1

EMBEDDINGS_FOLDER = "sentence_embeddings/"
files = os.listdir(EMBEDDINGS_FOLDER)
embeddings_matrix = np.zeros((len(files), vector_dimension[1]))

for iterator in xrange(0, len(files)):
    embeddings_matrix[iterator] = np.load(EMBEDDINGS_FOLDER+files[iterator])

def svm_function(testing_size):

    '''
    :param testing_size: Percentage of data to be used for testing.
    :return: Accuracy of the model
    '''

    X_train, X_test, y_train, y_test = train_test_split(embeddings_matrix, downsample.labels,
                                                        test_size=testing_size, random_state=0)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print score


def neural_networks(num_of_layers, nodes_per_layer, num_epochs,
                    activation_function, optimizer_function, loss_function):

    '''
    :param num_of_layers: Number of hidden layers to be used in deep neural network.
    :param nodes_per_layer: Number of nodes in each layer of the network.
    :param num_epochs: Number of epochs of training.
    :param activation_function: Activation function to be used. Default is relu function.
    :param optimizer_function: Optimizer to be used. Default is adam.
    :param loss_function: Loss function to be used. Default is binary_crossentropy.
    :return: Accuracy of the model.
    '''

    if activation_function == '':
        activation_function = 'relu'
    if optimizer_function == '':
        optimizer_function = 'adam'
    if loss_function == '':
        loss_function = 'categorical_crossentropy'

    model = Sequential()
    model.add(Dense(nodes_per_layer, input_dim=vector_dimension[1], activation='relu'))
    for _ in xrange(num_of_layers):
        model.add(Dense(nodes_per_layer, activation=activation_function))
    model.add(Dense(len(labels[0]), activation='sigmoid'))
    model.compile(loss=loss_function, optimizer=optimizer_function, metrics=['accuracy'])
    model.fit(embeddings_matrix, labels, epochs=num_epochs, batch_size=10, validation_split=test_size)
    scores = model.evaluate(embeddings_matrix, labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


if __name__ == "__main__":
    choice = input("Please choose the model to use:\n1. Neural Network\n2. SVM\nYour choice: ")
    if choice == 1:
        print neural_networks.__doc__
        print "Please specify the function parameters"
        layers = input("number_of_hidden_layers: ")
        nodes = input("nodes_per_layer: ")
        epochs = input("number_of_epochs: ")
        act_func = raw_input("activation_function (Press Enter for default): ")
        opt_func = raw_input("optimizer_function (Press Enter for default): ")
        loss_func = raw_input("loss_function (Press Enter for default): ")

        if not isinstance(layers, int) or not isinstance(epochs, int) \
                or not isinstance(nodes, int):
            print "Parameters not specified correctly :("
            sys.exit(-1)
        neural_networks(layers, nodes, epochs, act_func, opt_func, loss_func)
    elif choice == 2:
        print svm_function.__doc__
        print "Please specify the function parameters"
        testing_input_size = input("Testing input size: ")
        if not isinstance(testing_input_size, float) or testing_input_size > 1 or testing_input_size < 0:
            print "Parameters not specified correctly :("
            sys.exit(-1)
        svm_function(testing_input_size)
