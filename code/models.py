import sys
from collections import OrderedDict
import numpy

import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import layers,get_layer,dropout_layer
from utils import numpy_floatX

def init_params(options):
    params = OrderedDict()
    W = options['W']
    # embedding
    if W is None:
        randn = numpy.random.rand(options['n_words'],options['dim_proj'])
        params['Wemb'] = (0.01 * randn).astype(config.floatX)
    else:
        for i in range(len(W)):
            params['Wemb'+str(i)] = W[i].astype(config.floatX)

    if options['encoder'] == 'lstm':
        params = get_layer(options['encoder'])[0](options, params)

        # add deep feedforward layers
        deep = options['deep']
        for layer in range(deep):
            params['U'+str(layer+1)] = 0.01*numpy.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
            params['b'+str(layer+1)] = numpy.zeros((options['dim_proj'],)).astype(config.floatX)

        # classifier
        params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
        params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    elif options['encoder'] == 'cnnlstm' or options['encoder'] == 'cnngru':

        #gru or lstm layer parameters
        rnn = options['encoder'][3:]
        prefixs = []
        if not options['rnnshare']:
            for i in range(len(W)):
                for j in range(options['rnnlayer']):
                    layer_ij = rnn+'_emb'+str(i)+'_layer'+str(j)
                    prefixs.append(layer_ij)
                    if options['bidir']:
                        prefixs.append(layer_ij+'reverse')
        else:
            for j in range(options['rnnlayer']):
                layer_j = rnn+'_layer'+str(j)
                prefixs.append(layer_j)
                if options['bidir']:
                    prefixs.append(layer_j+'reverse')

        options['prefixs'] = prefixs
        #print prefixs

        for p in prefixs:
            params = get_layer(options['encoder'])[0](options,params,prefix=p)

        #print params.keys()

        if options['salstm'] and not options['bidir'] and not options['rnnshare'] and len(W) == 2 and options['rnnlayer'] == 1:
            #first emb is word2vec and then glove840b300d
            if options['dataset'] in ['sst5','sst2']:
                path = ['../sa_lstm/sst_word2vec_sa_lstm_model.npz','../sa_lstm/sst_glove840b300d_sa_lstm_model.npz']
            elif options['dataset'] == 'trec':
                path = ['../sa_lstm/trec_word2vec_sa_lstm_model.npz','../sa_lstm/trec_glove840b300d_sa_lstm_model.npz']

            cnt=0
            for pre,pa in zip(prefixs,path):
                #copy lstm parameters and Wemb
                pp = numpy.load(pa)
                for kk,vv in pp.iteritems():
                    if kk.startswith('lstm'):
                        #print pre+kk[4:]
                        params[pre+kk[4:]] = vv
                    elif kk.startswith('Wemb'):
                        #print kk
                        params['Wemb'+str(cnt)] == vv

                cnt += 1  
        #print params.keys()
         

        #cnn layer parameters
        params = get_layer(options['encoder'])[1](options,params)

        #classifer U,b
        feature_maps = options['feature_maps']
        filter_hs = options['filter_hs']
        params['U'] = 0.01 * numpy.random.randn(feature_maps*len(filter_hs),options['ydim']).astype(config.floatX)
        params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)
       
    else:
        print 'Undefined Encoder'

    return params

def build_model(tparams, options,SEED):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    projs = []
    for i in range(len(options['W'])):
        emb = tparams['Wemb'+str(i)][x.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])

        # LSTM or GRU Layer
        rnn = options['encoder'][3:]

        proj = emb
        for j in range(options['rnnlayer']):
            #Determine the prefix
            if options['rnnshare']:
                p = rnn+'_layer'+str(j)
            else:
                p = rnn+'_emb'+str(i)+'_layer'+str(j)
            #print p
            proj = get_layer(options['encoder'])[2](tparams, proj, options, mask=mask, prefix=p)
            proj = proj * mask[:,:,None]

        proj = proj.dimshuffle(0,1,2,'x')
        projs.append(proj)
        if options['combine']:
            emb = emb * mask[:,:,None]
            emb = emb.dimshuffle(0,1,2,'x')
            projs.append(emb)

    if options['bidir']:
        #reverse input
        x_flipud = x[::-1]
        mask_flipud = mask[::-1]
        for i in range(len(options['W'])):
            emb = tparams['Wemb'+str(i)][x_flipud.flatten()].reshape([n_timesteps,n_samples,options['dim_proj']])

            # LSTM or GRU Layer
            rnn = options['encoder'][3:]

            proj = emb
            for j in range(options['rnnlayer']):
                #Determine the prefix
                if options['rnnshare']:
                    p = rnn+'_layer'+str(j)+'reverse'
                else:
                    p = rnn+'_emb'+str(i)+'_layer'+str(j)+'reverse'
                #print p
                proj = get_layer(options['encoder'])[2](tparams, proj, options, mask=mask_flipud, prefix=p)
                proj = proj * mask_flipud[:,:,None]

            proj = proj.dimshuffle(0,1,2,'x')
            projs.append(proj)
            if options['combine']:
                emb = emb * mask_flipud[:,:,None]
                emb = emb.dimshuffle(0,1,2,'x')
                projs.append(emb)
    
    if options['encoder'] == 'lstm':
        # Average
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]

        # Feedforward
        for layer in range(options['deep']):
            #proj = tensor.nnet.relu(tensor.dot(proj, tparams['U'+str(layer+1)]) + tparams['b'+str(layer+1)])
            proj = tensor.dot(proj, tparams['U'+str(layer+1)]) + tparams['b'+str(layer+1)]
            #Relu
            proj = tensor.maximum(0.0,proj)

    elif options['encoder'] == 'cnnlstm' or options['encoder'] == 'cnngru':
        proj = tensor.concatenate(projs,3)
        # CNN Layer
        proj = get_layer(options['encoder'])[3](tparams,proj,options)

    else:
        print 'Undefined Encoder'


    # Dropout at the Penultimate layer
    if options['dropout_penul'] > 0:
        proj = dropout_layer(proj, use_noise, trng, dropout_rate=options['dropout_penul'])
    
    # Softmax
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


