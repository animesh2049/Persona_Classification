from downsample import *
import numpy as np
import pickle
import re
import os


import skipthoughts
import vocab
import train
import tools

X = []

#Make List of all sentences from all documents
for directory in data_label :
    files = os.listdir(DATA_DIR+directory)
    for f in files :
        file_descriptor = open(DATA_DIR+directory+"/"+f)
        file_content = file_descriptor.read()
        file_content = file_content.split('.')
        for sentence in file_content:
        #     sentence = re.split(r'[^a-zA-Z0-9]', sentence) # Check if it works
	        X.append(sentence)


worddict, wordcount = vocab.build_dictionary(X)
vocab.save_dictionary(worddict, wordcount, loc) #loc where you want to save dictionary


#in train.py set 1>path for dictionary, 2>save_to -path where to save model 3>maxlen_w
train.trainer(X)

#In tools.py set path_to_model=save_to in train, path_to_dictionary=dictionary in train and path_to_word2vec.
embed_map = tools.load_googlenews_vectors()
model = tools.load_model(embed_map)


for directory in data_label :
    files = os.listdir(DATA_DIR+directory)
    for f in files :
        file_descriptor = open(DATA_DIR+directory+"/"+f)
        file_content = file_descriptor.read()
        file_content = file_content.split('.')
        # for sentence in file_content:
        #     sentence = re.split(r'[^a-zA-Z0-9]', sentence) # Check if it works

        vectors = tools.encode(model,file_content) # It will give embedding of all the sentences of the document
        document_embedding = np.average(vectors, axis=0)
        document_embeddings.append(document_embedding)
        labels.append(label_to_id[directory])


document_embeddings_file = open('document_embeddings_from_skipthoughts.pickle', 'w')
labels_file = open('labels_from_skipthoughts.pickle', 'w')
pickle.dump(document_embeddings, document_embeddings_file)
pickle.dump(labels, labels_file)


