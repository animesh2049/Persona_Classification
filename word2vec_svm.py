from downsample import *
import numpy as np

GLOVE_DIR = "./"
GLOVE_FILE_NAME = "glove.840B.300d.txt"
glove = open(GLOVE_DIR + GLOVE_FILE_NAME)

word_embedding = {}
name_to_doc_id = {}
doc_id = 0

document_embedings = []
vector_dimension= 300

for line in glove:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embedding[word] = coefs

print "done loading embedding"
for directory in data_label:
    files = os.listdir(DATA_DIR+directory)
    for f in files:
        name_to_doc_id[f] = doc_id
        doc_id += 1
        file_descriptor = open(DATA_DIR+directory+"/"+f)
        temp_doc_embedding = np.zeros(vector_dimension)
        temp_word_counter = 0
        for word in file_descriptor:
            if word not in word_embedding:
                continue
            else:
                temp_doc_embedding += word_embedding[word]
                temp_word_counter += 1

        final_doc_embedding  = temp_doc_embedding/temp_word_counter
        document_embedings.append(final_doc_embedding)
