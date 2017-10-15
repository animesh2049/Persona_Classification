from downsample import *
import numpy as np
import pickle
import re

GLOVE_DIR = "./"
GLOVE_FILE_NAME = "glove.840B.300d.txt"
glove = open(GLOVE_DIR + GLOVE_FILE_NAME)

word_embedding = {}
name_to_doc_id = {}
doc_id = 0
document_embedings = []
labels = []
vector_dimension= 300

for line in glove:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embedding[word] = coefs

for directory in data_label :
    files = os.listdir(DATA_DIR+directory)
    for f in files :
        name_to_doc_id[f] = doc_id
        doc_id += 1
        file_descriptor = open(DATA_DIR+directory+"/"+f)
        temp_word_counter = 0
        final_doc_embedding = np.zeros(vector_dimension)
        file_content = file_descriptor.read()
        words = re.split(r'[^a-zA-Z0-9]', file_content)
        for word in words :
            if word :
                if word in word_embedding :
                    final_doc_embedding += word_embedding[word]
                    temp_word_counter += 1

        if temp_word_counter != 0 :
            final_doc_embedding  = final_doc_embedding / temp_word_counter

        document_embedings.append(final_doc_embedding)
        labels.append(label_to_id[directory])

print len(document_embedings)

document_embeddings_file = open('document_embeddings.pickle', 'w')
labels_file = open('labels.pickle', 'w')
pickle.dump(document_embedings, document_embeddings_file)
pickle.dump(labels, labels_file)
