# Code is based on https://github.com/cosmmb/DCNN/blob/master/process_TREC.py

import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(file, split_dict, label_dict, clean_string=False):
    """
    Loads data and split data
    """
    revs = []
    f = open(file)
    vocab = defaultdict(float)
    
    for index, line in enumerate(f.readlines()):       
        rev = []
        rev.append(line.strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev)
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum  = {"y":label_dict[index], 
                    "text": orig_rev,                             
                    "num_words": len(orig_rev.split()),
                    "split": split_dict[index]}
        revs.append(datum)

    return revs, vocab

 
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def get_labels(file):
    f = open(file)
    dict = {}
    for index,i in enumerate(f.readlines()):
        #index,l = i.strip().split('|')
        #index = int(index)
        dict[index] = int(i)   
    return dict

def get_split2(size=5952):
    
    dict = {}
    for i in range(size):
        if i < 5452:
            dict[i] = 1
        else:
            dict[i] =2

    return dict

def load_we_trec(W,word_idx_map,glove_file):
    #
    # Load GloVe Embeddings for TREC dataset
    #

    # Load Glove Embeddings
    vec_file = open(glove_file,'r')
    glove_vocab = {}
    for line in vec_file:
        word_em = line.split()
        word = word_em[0]
        embedding = np.array(word_em[1:]).astype(float)
        glove_vocab[word] = embedding

    # construct the We matrix
    trec_gloveWe = np.zeros(shape=W.shape)
    for word in word_idx_map.keys(): 
        idx = word_idx_map[word]
        # if the word is not in GloVe, use W instead
        if word in glove_vocab:
            trec_gloveWe[idx] = glove_vocab[word]
        else:
            trec_gloveWe[idx] = W[idx]

    return trec_gloveWe

if __name__=="__main__":
    save_path = "../../data/trec/trec.p"
    w2v_file = "../../data/GoogleNews-vectors-negative300.bin"   
    glove_file = "../../data/glove.840B.300d.txt"
    sent_file = "../../data/trec/TREC_all.txt"
    label_file = "../../data/trec/label_all.txt"
    label_dict = get_labels(label_file)
    
    split_dict = get_split2(5952)    
    
    print "loading data...",
    revs, vocab = build_data_cv(sent_file, split_dict, label_dict, \
                                clean_string=False)
    
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)

    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)

    print "loading GloVe vectors...",
    W2 = load_we_trec(W,word_idx_map,glove_file)
    print "GloVe loaded!"

    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W3, _ = get_W(rand_vecs)

    cPickle.dump([revs, W, W2, W3, word_idx_map, vocab], open(save_path, "wb"))
    print "dataset created!"
    
