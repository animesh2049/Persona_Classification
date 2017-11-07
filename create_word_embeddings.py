import re
import os
import pickle
import numpy as np

FILES = []
doc_id = 0
GLOVE_DIR = "./"
word_embedding = {}
vector_dimension = 300
name_to_doc_id_map = {}
GLOVE_FILE_NAME = "glove.840B.300d.txt"
DOCUMENT_WORD_EMBEDDINGS_FILE = "document_word_embeddings"
DATASET_FOLDERS = ["./blogs/orig", "./blogs/distant"]

for folder in DATASET_FOLDERS:
    directories = os.listdir(folder)
    for directory in directories:
        files = os.listdir(folder+"/"+directory)
        for f in files:
            FILES.append(folder+"/"+directory+"/"+f)

document_embeddings = np.empty([len(FILES), vector_dimension])

with open(GLOVE_DIR + GLOVE_FILE_NAME) as glove:
    for line in glove:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embedding[word] = coefs

for f in FILES:
    with open(f) as file_descriptor:
        temp_word_counter = 0
        final_doc_embedding = np.zeros(vector_dimension)
        file_content = file_descriptor.read()
        words = re.split(r'[^a-zA-Z0-9]', file_content)
        for word in words:
            if word:
                if word in word_embedding:
                    final_doc_embedding += word_embedding[word]
                    temp_word_counter += 1

        if temp_word_counter != 0:
            final_doc_embedding = final_doc_embedding / temp_word_counter

        document_embeddings[doc_id] = final_doc_embedding
    doc_id += 1

np.save(DOCUMENT_WORD_EMBEDDINGS_FILE, document_embeddings)
