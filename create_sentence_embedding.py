import re
import pickle
import numpy as np
import skipthoughts
from downsample import *
from nltk import sent_tokenize

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
document_embeddings = open('document_embeddings_from_skipthoughts', 'w')

for directory in directory_to_file_map:
    files = directory_to_file_map[directory]
    for f in files:
        print DATA_DIR+directory+"/"+f
        with open(DATA_DIR+directory+"/"+f) as file_descriptor:
            file_content = sent_tokenize(file_descriptor.read())
            document_embedding = np.average(encoder.encode(file_content), axis=0)
            document_embeddings.write(','.join(str(e) for e in document_embedding))

document_embeddings.close()
