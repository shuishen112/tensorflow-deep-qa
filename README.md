This is a tensorflow implementation of  paper

[Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf)

I study code from these people

[cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)

[pair-cnn-ranking](https://github.com/zhangzibin/PairCNN-Ranking)

[gan-for-qa](https://github.com/wabyking/GAN-for-QA)

## Requirements

-python2.7

-Tensorflow > 0.12

-gensim

-numpy

## Training


```
./train.py
```

##

now the result is silly because i don't add the overlap feattures

-trec

map | mrr

---- | ----

0.65 | 0.65

-wiki(clean)

map | mrr

0.656 | 0.656

i will add overlap soon



