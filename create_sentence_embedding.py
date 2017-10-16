from downsample import *
import numpy as np
import pickle
from nltk import sent_tokenize
import re
import skipthoughts

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
# document_embeddings = []
document_embeddings = open('document_embeddings_from_skipthoughts', 'w')

for directory in data_label :
    files = os.listdir(DATA_DIR+directory)
    for f in files :
        file_descriptor = open(DATA_DIR+directory+"/"+f)
        file_content = file_descriptor.read()
        file_content = sent_tokenize(file_content)
        vectors = encoder.encode(file_content) # It will give embedding of all the sentences of the document
        document_embedding = np.average(vectors, axis=0)
        s = ""
        for doc in document_embedding :
            s += str(doc) + ","
        s = s[:-1]
        document_embeddings.write(s)
        # document_embeddings.append(document_embedding)

# document_embeddings_file = open('document_embeddings_from_skipthoughts.pickle', 'w')
# pickle.dump(document_embeddings, document_embeddings_file)
