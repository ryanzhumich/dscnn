#! /bin/bash

THEANO_FLAGS=device=gpu0,floatX=float32 python dscnn.py \
    -dataset trec \
    -validportion 0.15 \
    -batchsize 8 \
