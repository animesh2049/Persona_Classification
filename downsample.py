import os
import numpy as np
from keras.utils import to_categorical

DATA_DIR = "blogs/mod/"
directories = os.listdir(DATA_DIR) # Consultant, Patient etc.
temp_map = {}
directory_to_file_map = {}
files_to_ignore = {}
label_to_id_map = {}
directory_id = 0
file_counter = 0
labels = []

for directory in directories:
    files = os.listdir(DATA_DIR+directory)  # Files in inside consultant folder
    for f in files:
        if f in temp_map:
            files_to_ignore[f] = 1
        else:
            temp_map[f] = 1

for directory in directories:
    label_to_id_map[directory] = directory_id
    directory_id += 1
    directory_to_file_map[directory] = []
    files = os.listdir(DATA_DIR+directory)
    for f in files:
        if f not in files_to_ignore:
            directory_to_file_map[directory].append(f)
            labels.append(directory_id-1)
            file_counter += 1

categorial_labels = to_categorical(np.asarray(labels))
np.save("labels", categorial_labels)
