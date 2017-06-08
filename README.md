This is a tensorflow implementation of  paper

[Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf)

I study code from these people

[cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)

[pair-cnn-ranking](https://github.com/zhangzibin/PairCNN-Ranking)

[gan-for-qa](https://github.com/wabyking/GAN-for-QA)

the embedding can be download from :

[https://github.com/aseveryn/deep-qa](https://github.com/aseveryn/deep-qa)

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

now the result is silly because i don't add the overlap features

and i don't think the similarity layer promote the result

| data | map | mrr |
| :--- | :----: | ----: |
| trec | 0.65 | 0.65 |
| wiki(clean) | 0.658 | 0.67  |

i will add overlap soon

when a add overlap feature as embedding

we can get the result

| data | map | mrr |
| :--- | :----: | ----: |
| trec | 0.747 | 0.74 |
| wiki(clean) | 0.68 | 0.68  |



