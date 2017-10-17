import re
import gc
import pickle
import numpy as np
import skipthoughts
from downsample import *
from nltk import sent_tokenize

count = 0
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

for directory in data_label:
    files = data_label[directory]
    for f in files:
        count += 1
        print DATA_DIR+directory+"/"+f, count
        with open(DATA_DIR+directory+"/"+f) as file_descriptor:
            file_content = sent_tokenize(file_descriptor.read())
            document_embedding = np.average(encoder.encode(file_content, verbose=False), axis=0)
            np.save("document_embeddings/" + f, document_embedding)
        if count % 10 == 0:
            gc.collect()

document_embeddings.close()
