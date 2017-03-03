import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

from utils import norm_weight,ortho_weight,_p,ReLU,numpy_floatX

def get_layer(name):
    fns = layers[name]
    return fns

# CNN layer
def param_init_cnn(options,params,prefix='cnn'):
    feature_maps = options['feature_maps']
    filter_hs = options['filter_hs']

    # Fixed image shape
    num_chn = len(options['W'])
    if options['bidir']:
        num_chn = num_chn * 2

    if options['combine']:
        num_chn = num_chn * 2

    image_shape = (options['batch_size'],num_chn,options['maxlen'],options['dim_proj'])
    img_h = image_shape[2]
    img_w = image_shape[3]
    options['image_shape'] = image_shape

    # init filter,bias
    filter_shapes = []
    pool_sizes = []

    filter_w = options['dim_proj']
    for filter_h in filter_hs:
        filter_shape = (feature_maps,num_chn,filter_h,filter_w)
        pool_size = (img_h-filter_h+1,img_w-filter_w+1)

        #4 different initialization of filters
        if options['init'] == 'uniform':
            params['cnn_f'+str(filter_h)] = numpy.random.uniform(low=-0.01,high=0.01,size=filter_shape).astype(config.floatX)
        elif options['init'] == 'xavier':
            fan_in = numpy.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(pool_size))
            W_bound = numpy.sqrt(6. /(fan_in + fan_out))
            params['cnn_f'+str(filter_h)] = numpy.random.uniform(low=-W_bound,high=W_bound,size=filter_shape).astype(config.floatX)
        elif options['init'] == 'gaussian':
            params['cnn_f'+str(filter_h)] = numpy.random.normal(size=filter_shape).astype(config.floatX)
        elif options['init'] == 'ortho':
            W_ortho = ortho_weight(numpy.prod(filter_shape[1:]))
            W_ortho = numpy.reshape(W_ortho[:filter_shape[0]],filter_shape)
            params['cnn_f'+str(filter_h)] = W_ortho
        
        params['cnn_b'+str(filter_h)] = numpy.zeros((filter_shape[0],)).astype(config.floatX)

        filter_shapes.append(filter_shape)
        pool_sizes.append(pool_size)

    options['filter_shapes'] = filter_shapes
    options['pool_sizes'] = pool_sizes
   
    
    return params

def cnn_layer(tparams,proj,options):
    #proj = proj.dimshuffle(1,'x',0,2)  #(batchsize,1,max_len,dim_proj)
    proj = proj.dimshuffle(1,3,0,2)  # (maxlen,n_sample(batchsize), dim_proj, num_chn) -> (batchsize,num_chn,max_len,dim_proj)

    #image_shape = proj.shape
    filter_shapes = options['filter_shapes']
    image_shape = options['image_shape']
    pool_sizes = options['pool_sizes']

    image_shape = (None,image_shape[1],image_shape[2],image_shape[3])

    conv_outs = []
    for i in range(len(filter_shapes)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        
        #img_h = image_shape[2]
        filter_h = filter_shape[2]
        #img_w = image_shape[3]
        #filter_w = filter_shape[3]
        #poolsize = (img_h-filter_h+1,img_w-filter_w+1)   

        conv_out = conv.conv2d(input=proj,filters=tparams['cnn_f'+str(filter_h)],filter_shape=filter_shape,image_shape=image_shape)
        conv_out_relu = ReLU(conv_out + tparams['cnn_b'+str(filter_h)].dimshuffle('x',0,'x','x'))
        if options['pool_type'] == 'max':
            conv_out_pool = pool.pool_2d(input=conv_out_relu,ds=pool_size,ignore_border=True,mode='max')
        elif options['pool_type'] == 'avg':
            conv_out_pool = conv_out_relu.flatten(3)
            conv_out_pool = tensor.mean(conv_out_pool,axis=2)
        else:
            sys.exit()
        conv_outs.append(conv_out_pool.flatten(2))
    proj = tensor.concatenate(conv_outs,1)

    return proj 

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    """
    Gated Recurrent Unit (GRU)
    """
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']

    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype(config.floatX)

    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype(config.floatX)

    return params

def gru_layer(tparams, state_below, options, init_state=None, prefix='gru', mask=None, **kwargs):
    """
    Feedforward pass through GRU
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [init_state],
                                non_sequences = [tparams[_p(prefix, 'U')],
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=False,
                                strict=True)
    #rval = [rval]
    return rval

#LSTM layer
def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    # outputs_info include h_ and c_
    # return only hidden states, so return rval[0]
    return rval[0]

def dropout_layer(state_before, use_noise, trng,dropout_rate):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=dropout_rate, n=1,
                                        dtype=state_before.dtype)),
                         state_before * dropout_rate)
    return proj

layers = {'lstm': (param_init_lstm, 'spaceholder', lstm_layer)}
layers['cnnlstm'] = (param_init_lstm, param_init_cnn,lstm_layer,cnn_layer)
layers['cnngru'] = (param_init_gru, param_init_cnn, gru_layer,cnn_layer)


