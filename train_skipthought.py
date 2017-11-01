from downsample import *
import numpy as np
import re
import pdb
import os
import skipthoughts
import vocab
import train
import tools
from nltk import sent_tokenize

X = []
file_counter = 0
SENTENCE_EMBEDDING_FOLDER = "sentence_embeddings2/"
path_to_word2vec = "/home/aayush/Persona_Classification/GoogleNews-vectors-negative300.bin"

choice = input("Please choose the model to use:\n1. Train sentences on words trained on google news vectors\
\n2. Train sentences on words on trained on glove vectors\nYour choice: ")

if choice == 2:
    SENTENCE_EMBEDDING_FOLDER = "sentence_embeddings3/"
    path_to_word2vec = "/home/aayush/Persona_Classification/glove.840B.300d.w2vformat.txt"

#Make List of all sentences from all documents
for directory in directory_to_file_map:
    files = os.listdir(DATA_DIR+directory)
    for f in files:
        file_counter += 1
        file_descriptor = open(DATA_DIR+directory+"/"+f)
        file_content = sent_tokenize(file_descriptor.read())
        for sentence in file_content:
            if sentence:
                X.append(sentence)

sentence_embeddings = np.empty([file_counter, 4800])
loc = "/home/aayush/Persona_Classification/dictionary.pkl"
saveto = "/home/aayush/Persona_Classification/toy.npz"
maxlen_w = 70
worddict, wordcount = vocab.build_dictionary(X)
vocab.save_dictionary(worddict, wordcount, loc) #loc where you want to save dictionary

#in train.py set 1>path for dictionary, 2>save_to -path where to save model 3>maxlen_w
train.trainer(X, dictionary=loc, saveto=saveto, maxlen_w=maxlen_w)

#In tools.py set path_to_model=save_to in train, path_to_dictionary=dictionary in train and path_to_word2vec.
count = 0
embed_map = tools.load_googlenews_vectors(path_to_word2vec)
model = tools.load_model(embed_map)

for directory in directory_to_file_map:
    files = os.listdir(DATA_DIR+directory)
    for f in files:
        count += 1
        print DATA_DIR+directory+"/"+f, count
        with open(DATA_DIR+directory+"/"+f) as file_descriptor:
            file_content = sent_tokenize(file_descriptor.read())
            document_embedding = tools.encode(model, file_content, verbose=False)
            document_embedding = np.average(document_embedding, axis=0)
            np.save(SENTENCE_EMBEDDING_FOLDER+f[:-4], document_embedding)
