#-*- coding: utf-8 -*-
from downsample import *
import numpy as np
import re
import os
from config import base_path_to_directory
import skipthoughts
import vocab
import train
import tools
from nltk import sent_tokenize

X = []
FILES = []
file_counter = 0
SENTENCE_EMBEDDING_FOLDER = "document_sentence_embeddings2/"
DATASET_FOLDERS = ["./blogs/orig", "./blogs/distant"]
path_to_word2vec = base_path_to_directory + "GoogleNews-vectors-negative300.bin"

choice = input("Please choose the model to use:\n1. Train sentences on words trained on google news vectors\
\n2. Train sentences on words on trained on glove vectors\nYour choice: ")

if choice == 2:
    SENTENCE_EMBEDDING_FOLDER = "document_sentence_embeddings3/"
    path_to_word2vec = base_path_to_directory + "glove.840B.300d.w2vformat.txt"

#Make List of all sentences from all documents
for folder in DATASET_FOLDERS:
    directories = os.listdir(folder)
    for directory in directories:
        files = os.listdir(folder+"/"+directory)
        for f in files:
            FILES.append(folder+"/"+directory+"/"+f)

for f in FILES:
    file_counter += 1
    with open(f) as file_descriptor:
        file_content = file_descriptor.read().decode("utf-8", "ignore")
        file_content = sent_tokenize(file_content)
        for sentence in file_content:
            if sentence:
                X.append(sentence.strip())

sentence_embeddings = np.empty([file_counter, 4800])
loc = base_path_to_directory + "dictionary.pkl"
saveto = base_path_to_directory + "toy.npz"
maxlen_w = 70
worddict, wordcount = vocab.build_dictionary(X)
vocab.save_dictionary(worddict, wordcount, loc) #loc where you want to save dictionary
#in train.py set 1>path for dictionary, 2>save_to -path where to save model 3>maxlen_w
train.trainer(X, dictionary=loc, saveto=saveto, maxlen_w=maxlen_w)

#In tools.py set path_to_model=save_to in train, path_to_dictionary=dictionary in train and path_to_word2vec.
embed_map = tools.load_googlenews_vectors(path_to_word2vec)
model = tools.load_model(embed_map)
if not os.path.exists(SENTENCE_EMBEDDING_FOLDER):
    os.mkdir(SENTENCE_EMBEDDING_FOLDER)

for f in FILES:
    with open(f) as file_descriptor:
        file_content = sent_tokenize(file_descriptor.read())
        document_embedding = tools.encode(model, file_content, verbose=False)
        document_embedding = np.average(document_embedding, axis=0)
        file_name = f.split('/')[-1]
        np.save(SENTENCE_EMBEDDING_FOLDER+file_name[:-4], document_embedding)
