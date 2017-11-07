import os
import numpy as np
import skipthoughts
from nltk import sent_tokenize

FILES = []
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
SENTENCE_EMBEDDING_FOLDER = "document_sentence_embeddings/"
if not os.path.exists(SENTENCE_EMBEDDING_FOLDER):
    os.mkdir(SENTENCE_EMBEDDING_FOLDER)
DATASET_FOLDERS = ["./blogs/orig", "./blogs/distant"]

for folder in DATASET_FOLDERS:
    directories = os.listdir(folder)
    for directory in directories:
        files = os.listdir(folder+"/"+directory)
        for f in files:
            FILES.append(folder+"/"+directory+"/"+f)

for f in FILES:
    with open(f) as file_descriptor:
        file_content = sent_tokenize(file_descriptor.read())
        document_embedding = encoder.encode(file_content, verbose=False)
        document_embedding = np.average(document_embedding, axis=0)
        file_name = f.split('/')[-1]
        np.save(SENTENCE_EMBEDDING_FOLDER+file_name[:-4], document_embedding)
