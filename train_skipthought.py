from downsample import *
import numpy as np
import re
import os
import skipthoughts
import vocab
import train
import tools
from nltk import sent_tokenize

X = []
sentence_embeddings = np.empty([350, 2400])

#Make List of all sentences from all documents
for directory in directory_to_file_map :
    files = os.listdir(DATA_DIR+directory)
    for f in files :
        file_descriptor = open(DATA_DIR+directory+"/"+f)
        file_content = sent_tokenize(file_descriptor.read())
        for sentence in file_content:
            if sentence:
                X.append(sentence)

X = X[:10]
loc = "/home/aayush/Persona_Classification/dictionary.pkl"
saveto = "/home/aayush/Persona_Classification/toy.npz"
maxlen_w = 70
worddict, wordcount = vocab.build_dictionary(X)
vocab.save_dictionary(worddict, wordcount, loc) #loc where you want to save dictionary

#in train.py set 1>path for dictionary, 2>save_to -path where to save model 3>maxlen_w
train.trainer(X, dictionary=loc, saveto=saveto, maxlen_w=maxlen_w, max_epochs=2)

#In tools.py set path_to_model=save_to in train, path_to_dictionary=dictionary in train and path_to_word2vec.
embed_map = tools.load_googlenews_vectors()
model = tools.load_model(embed_map)
count = 0

for directory in directory_to_file_map :
    files = os.listdir(DATA_DIR+directory)
    for f in files :
        file_content = sent_tokenize(file_descriptor.read())
        document_embedding = encoder.encode(file_content, verbose=False)
        document_embedding = np.average(document_embedding, axis=0)
        document_embedding = document_embedding[::2]
        for i in range(4800):
            sentence_embeddings[count][i] = document_embedding[i]
        count += 1
        if count % 20 == 0:
            break
    if count % 20 == 0:
        break

np.save('trained_sentence_embeddings', sentence_embeddings)
