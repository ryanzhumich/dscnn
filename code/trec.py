import numpy
import theano
import cPickle

def prepare_data(seqs, labels, maxlen=None,pad=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l <= maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)

    if maxlen is None:
        maxlen = numpy.max(lengths)

    # here we can also implement pad
    if pad is not None:
        maxlen = maxlen+2*pad
        x = numpy.zeros((maxlen, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
        for idx,s in enumerate(seqs):
            x[pad:pad + lengths[idx],idx] = s 
            x_mask[pad:pad + lengths[idx],idx] = 1.

    elif pad is None:
        x = numpy.zeros((maxlen, n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels

def get_idx_from_sent(sent,word_idx_map):
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    return x


def load_data(revs,word_idx_map,sort_by_len=True,valid_portion=0.1):
    train_set_x = []
    train_set_y = []
    valid_set_x = []
    valid_set_y = []
    test_set_x = []
    test_set_y = []
 
    # construct train and test dataset
    for rev in revs:
        sent = get_idx_from_sent(rev["text"],word_idx_map)
        
        if rev["split"] == 2:
            test_set_x.append(sent)
            test_set_y.append(rev["y"])
        else:
            train_set_x.append(sent)
            train_set_y.append(rev["y"])

    print 'Original Fold:','Train',len(train_set_x), 'Test', len(test_set_x)

    # reserve some validation data
    n_samples = len(train_set_x)

    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1.-valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]

    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    print 'Researve Validation:','Train',len(train_set_x), 'Test', len(test_set_x), 'Valid', len(valid_set_x)

    # sort by length
    def len_argsort(seq):
        return sorted(range(len(seq)), key = lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]
         
        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
         

    train = (train_set_x, train_set_y)
    valid = (valid_set_x,valid_set_y) 
    test = (test_set_x, test_set_y)

    return train,valid,test

