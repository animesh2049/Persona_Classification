from downsample import *
import numpy as np
import pickle
import re
import skipthoughts

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
document_embeddings = []
labels = []

for directory in data_label :
    files = os.listdir(DATA_DIR+directory)
    for f in files :
        file_descriptor = open(DATA_DIR+directory+"/"+f)
        file_content = file_descriptor.read()
        file_content = file_content.split('.')
        for sentence in file_content:
            sentence = re.split(r'[^a-zA-Z0-9]', sentence) # Check if it works

        vectors = encoder.encode(file_content) # It will give embedding of all the sentences of the document
        document_embedding = np.average(vectors, axis=0)
        document_embeddings.append(document_embedding)
        labels.append(label_to_id[directory])


document_embeddings_file = open('document_embeddings_from_skipthoughts.pickle', 'w')
labels_file = open('labels_from_skipthoughts.pickle', 'w')
pickle.dump(document_embeddings, document_embeddings_file)
pickle.dump(labels, labels_file)
