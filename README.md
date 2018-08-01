# Question answering 

## Motivation
The purpose of this repository is to explore question answering model in deep learning. We want to build a benchmark for question answering tasks by using the tensorflow API [Dataset](https://www.tensorflow.org/guide/datasets) and [Estimator](https://www.tensorflow.org/guide/estimators) which can help us import the dataset in parallel.


<!-- 这是一个处理sentence pairs,question answering问题的工具包
在trecqa数据集上overlap或者说是word count这个特征非常重要，注意去停用词和不去停用词效果差异很大，不去停用词
map mrr = (0.68,0.73)

对于直接unigram embedding(未去停用词)，map,mrr = (0.59,0.65)
去停用词 map,mrr(0.62,0.67)
map,mrr = (0.71,0.77) 一般可以提升2个点左右，所以这个提升是非常大的。 -->

## Models

- [Deep Learning for Answer Sentence Selection](https://arxiv.org/abs/1412.1632)

- [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](http://disi.unitn.it/~severyn/papers/sigir-2015-long.pdf)

- [Attentive Pooling Networks](https://arxiv.org/abs/1602.03609)

- [Inner Attention based Recurrent Neural Networks for Answer Selection](http://www.aclweb.org/anthology/P16-1122)

## Requirements

-python2.7

-Tensorflow = 1.8

## Processing


```
./run.py --task_type prepare
```

## training
```
./run.py --task_type train
```

## test
```
./run.py --task_type infer
```

## Contributor

-   [@ZhanSu](https://github.com/shuishen112)
-   [@Wabywang](https://github.com/Wabyking)
<!-- now the result is silly because i don't add the overlap features -->

<!-- and i don't think the similarity layer promote the result


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
| wiki(clean) | 0.687 | 0.708  | -->



