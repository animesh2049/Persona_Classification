import os

DATA_DIR = "blogs/mod/"

directories = os.listdir(DATA_DIR) # Consultant, Patient etc.
temp_map = {}
files_to_ignore = {}
label_to_id = {}
temp_id_counter = 0

for directory in directories:
    files = os.listdir(DATA_DIR+directory)  # Files in inside consultant folder
    for f in files:
        if f in temp_map:
            files_to_ignore[f] = 1
        else:
            temp_map[f] = 1

data_label = {}

for directory in directories:
    label_to_id[directory] = temp_id_counter
    temp_id_counter += 1
    data_label[directory] = []
    files = os.listdir(DATA_DIR+directory)
    for f in files:
        if f not in files_to_ignore:
            data_label[directory].append(f)