import numpy as np
import skipthoughts
from downsample import *
from nltk import sent_tokenize

count = 0
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
SENTENCE_EMBEDDING_FOLDER = "sentence_embeddings/"

for directory in directory_to_file_map:
    files = directory_to_file_map[directory]
    for f in files:
        count += 1
        print DATA_DIR+directory+"/"+f, count
        with open(DATA_DIR+directory+"/"+f) as file_descriptor:
            file_content = sent_tokenize(file_descriptor.read())
            document_embedding = encoder.encode(file_content, verbose=False)
            document_embedding = np.average(document_embedding, axis=0)
            np.save(SENTENCE_EMBEDDING_FOLDER+f[:-4], document_embedding)
