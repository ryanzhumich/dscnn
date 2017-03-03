import time,sys
import numpy
import theano
from theano import config
import theano.tensor as tensor

from models import init_params,build_model
from optimizers import optims
from utils import numpy_floatX,get_minibatches_idx,load_params,init_tparams,unzip,zipp

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index])
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs

def pred_error(f_pred, prepare_data, data, iterator, fname='', verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    if verbose:
        f = open('../data/trec/TREC_10.label')
        lines = f.readlines()
        f.close()
        f_out = open(fname+'trec_out_dscnn.txt','w')
        cat_ind = ['ABBR','ENTY','DESC','HUM','LOC','NUM']
        cnt = 0
    for b, valid_index in iterator:
        x ,mask,y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index])
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        
        
        if verbose:
            for i in range(len(preds)):
                p = preds[i]
                if p != targets[i]:
                    f_out.write('*')
                f_out.write(cat_ind[p] + ' ')
                f_out.write(lines[cnt])
                cnt += 1       
        
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err * 100


def train_model(
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=100,  # The maximum number of epoch to run
    dispFreq=500,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer='adadelta', 
    encoder='lstm', 
    rnnshare=True,
    bidir=False,
    saveto=None,  # The best model will be saved there
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',
    W = None, # embeddings
    deep = 0, # number of layers above
    rnnlayer = 0, # number of rnn layers
    filter_hs = [3,4,5], #filter's width
    feature_maps = 100,  #number of filters
    pool_type = 'max',    #pooling type
    combine = False,
    init='uniform',
    salstm=False,
    noise_std=0.,
    dropout_penul=0.5, 
    reload_model=None,  # Path to a saved model we want to start from.
    data_loader=None,
    fname='',
):
    

    # Model options
    optimizer = optims[optimizer]
    model_options = locals().copy()
    #print "model options", model_options


    # Load data
    (load_data, prepare_data) = data_loader

    print 'Loading',dataset,'data'
    train,valid,test = load_data()

    ydim = numpy.max(train[1]) + 1
    model_options['ydim'] = ydim

     
    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params(reload_model, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options,SEED)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.

        weight_decay += (tparams['U'] ** 2).sum()

        if model_options['encoder'] == 'lstm':
            for layer in range(model_options['deep']):
                weight_decay += (tparams['U'+str(layer+1)] ** 2).sum()
        elif model_options['encoder'] == 'cnnlstm':
            for filter_h in model_options['filter_hs']:
                weight_decay += (tparams['cnn_f'+str(filter_h)] ** 2).sum()
        
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, x, mask, y, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'
                
                
                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)
                    history_errs.append([valid_err, test_err])

                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)

                    if ((len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min())):
                        bad_counter += 1
                        
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test, fname=fname, verbose=True)

    print 'Train Error', train_err, 'Valid Error', valid_err, 'Test Error', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' % (end_time - start_time))
    return train_err, valid_err, test_err

