# Persona Classification
Medical social media is a subset of social media that is restricted to health care related topics. Different kinds of people (or personae) contribute to the medical social media - like patients, caretakers, consultants, pharmacists, medical researchers or medical journalists. The problem at hand is that for a given a blog post as an input, the system is expected to return which personae wrote that post.

## Dependencies
This code is written in python. To use it you will need:

* Python 2.7
* [skip-thoughts](https://github.com/ryankiros/skip-thoughts)
* [glove-vectors-web-crawl-2.0GB](https://nlp.stanford.edu/projects/glove/)

## Getting started

You will first need to download the model files, word embeddings and blog posts data (see below). The embedding files (utable and btable) are quite large (>2GB) so make sure there is enough space available. The encoder vocabulary can be found in dictionary.txt.

wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl

## Dataset

* [blog-posts-dataset](https://drive.google.com/file/d/0B_9ISEpIrWxEVGw4aGttWTFGT0U/view)
