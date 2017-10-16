import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

EMBEDDING_FILE_NAME = "document_embeddings" # For other cases this can also be taken as input
LABEL_FILE_NAME = "labels"

embeddng_file = open(EMBEDDING_FILE_NAME, 'r')
label_file = open(LABEL_FILE_NAME, 'r')
vector_dimension = 300
document_embedings = pickle.load(embeddng_file)
labels = pickle.load(label_file)
test_size = 0.1
document_embedings = np.array(document_embedings)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(document_embedings, labels, test_size = test_size, random_state = 0)
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
        loss_function = 'crossentropy'

    model = Sequential()
    model.add(Dense(nodes_per_layer, input_dim=vector_dimension, activation='relu'))
    for _ in xrange(num_of_layers):
        model.add(Dense(nodes_per_layer, activation=activation_function))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_function, optimizer=optimizer_function, metrics=['accuracy'])
    model.fit(document_embedings, labels, epochs=num_epochs, batch_size=10, validation_split=test_size)
    scores = model.evaluate(document_embedings, labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


if __name__ == "__main__":
    choice = raw_input("Please choose the model to use:\n1. Neural Network\n2. SVM\nYour choice:")
    if choice == 1:
        print neural_networks.__doc__
        print "Please specify the function parameters"
        layers = input("num_of_layers: ")
        nodes = input("nodes_per_layer: ")
        epochs = input("num_epochs: ")
        act_func = raw_input("activation_function (Press Enter for default): ")
        opt_func = raw_input("optimizer_function (Press Enter for default): ")
        loss_func = raw_input("loss_function (Press Enter for default): ")

        if not isinstance(layers, int) or not isinstance(epochs, int) \
                or not isinstance(nodes, int):
            print "Parameters not specified correctly :("
            exit(-1)
        neural_networks(layers, nodes, epochs, act_func, opt_func, loss_func)
