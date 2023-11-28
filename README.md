this was so far the best project i have work with.
# Dependency Sensitive Convolutional Neural Networks
Code for the paper [Dependency Sensitive Convolutional Neural Networks for Modeling Sentences and Documents](https://arxiv.org/abs/1611.02361) (NAACL 2016)

Demo with [TREC](http://cogcomp.cs.illinois.edu/Data/QA/QC) dataset

The implementation is based on https://github.com/yoonkim/CNN_sentence and http://deeplearning.net/tutorial/lstm.html

## Dependencies
- Python (2.7)
- Theano (0.8)
- Pandas (0.17)

## Prepare Pretrained Word Embeddings

The model uses preptrained word embeddings including [word2vec](https://code.google.com/archive/p/word2vec/) and [GloVe](http://nlp.stanford.edu/data/glove.840B.300d.zip).
Download those word embeddings and save them as:

- word2vec: data/GoogleNews-vectors-negative300.bin
- GloVe:    data/glove.840B.300d.txt

## Data Preprocessing

```
cd code/preprocess/
python process_trec.py
```

## Run Demo with Training and Testing

```
cd code/
./run_trec_demo.sh 
```


