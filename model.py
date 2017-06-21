#!/usr/bin/env python
# encoding: utf-8

__author__ = 'liming-vie'

import os
import sys

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

from tqdm import tqdm

import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_string('train_dir', '../train_data', 'directory of training data')
tf.app.flags.DEFINE_string('train_file', '../output/train.ids', '')
tf.app.flags.DEFINE_string('test_file', '../output/test.ids', '')
tf.app.flags.DEFINE_string('result_file', '../output/result', '')
tf.app.flags.DEFINE_string('word2vec_file', '../output/word2vec.txt', '')
tf.app.flags.DEFINE_string('vocab_file', '../output/vocab', '')
tf.app.flags.DEFINE_boolean('test', False, 'set to True for predict task')

tf.app.flags.DEFINE_string('ckpt_per_steps', 500, '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_float('init_lr', 0.0001, '')
tf.app.flags.DEFINE_float('lr_decay', 0.95, '')
tf.app.flags.DEFINE_float('l2_coef', 0.2, '')
tf.app.flags.DEFINE_integer('num_filters', 64,'')
tf.app.flags.DEFINE_string('filter_sizes', '1,2,3,4,5,6,7,8,9,10,15,20', '')
tf.app.flags.DEFINE_string('mlp_units', '512,512,256,64,32', '')

tf.app.flags.DEFINE_integer('embed_size', 128, '')
tf.app.flags.DEFINE_string('question_max_length', 30,'')
tf.app.flags.DEFINE_string('answer_max_length', 50,'')

FLAGS = tf.app.flags.FLAGS

class DBQA:
  def __init__(self):
    vocab2idx = self.load_vocab()
    self.vocab_size = len(vocab2idx)
    word2vec = self.load_word2vec(vocab2idx)

    filter_sizes = map(int, FLAGS.filter_sizes.split(','))
    mlp_units = map(int, FLAGS.mlp_units.split(','))

    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.Session(config=config)

    '''graph'''
    print 'Initializing graph...'

    with tf.variable_scope('embedding_layer'):
      self.embed_matrix = tf.Variable(word2vec, dtype=tf.float32,
        name='embedding_matrix')
      with tf.variable_scope('question'):
        self.question = tf.placeholder(tf.int32, shape=[None, None],
          name='question') # [batch_size, sequence_length]
        self.question_overlap = tf.placeholder(tf.float32, shape=[None, None],
          name='question_overlap')
        question_matrix = self.embed_with_overlap(self.question, self.question_overlap)
      with tf.variable_scope('answer'):
        self.answer = tf.placeholder(tf.int32, shape=[None, None],
          name='answer') # [batch_size, sequence_length]
        self.answer_overlap = tf.placeholder(tf.float32, shape=[None, None],
          name='answer_overlap')
        answer_matrix = self.embed_with_overlap(self.answer, self.answer_overlap)

    with tf.variable_scope('conv_pool_layer'):
      question_pooled, q_size = self.convolution_max_pool_layers(\
        question_matrix, FLAGS.question_max_length, filter_sizes, 'question')
      answer_pooled, a_size = self.convolution_max_pool_layers(\
        answer_matrix, FLAGS.answer_max_length, filter_sizes, 'answer')

    with tf.variable_scope('interact_layer'):
      M = tf.get_variable('similarity_matrix',\
        shape=[q_size, a_size],
        initializer=tf.truncated_normal_initializer())
      qM = tf.tensordot(question_pooled, M, 1)
      sim_score = tf.reduce_sum(qM * answer_pooled, axis=1, keep_dims=True)

    with tf.variable_scope('multi_perceptron_layer'):
      mlp_inputs = tf.concat([question_pooled, sim_score, answer_pooled], -1)
      mlp_inputs = tf.reshape(mlp_inputs, shape=[-1, q_size+a_size+1])
      for i, mlp_unit in enumerate(mlp_units):
        with tf.variable_scope('mlp_%d'%i):
          mlp_outputs = tf.contrib.layers.legacy_fully_connected(
            mlp_inputs, mlp_unit,
            activation_fn=tf.nn.relu,
            weight_regularizer=tf.contrib.layers.l2_regularizer(
              FLAGS.l2_coef))
          mlp_inputs = mlp_outputs

    with tf.variable_scope('dropout_layer'):
      self.training = tf.placeholder(tf.bool, name='is_training')
      dropout_out = tf.layers.dropout(mlp_outputs, training=self.training)

    with tf.variable_scope('output_layer'):
      W = tf.get_variable('W', shape=[mlp_units[-1], 2],
        initializer=tf.contrib.layers.xavier_initializer())
      b = tf.get_variable('b', shape=[2],
        initializer=tf.random_normal_initializer())
      logits = tf.nn.xw_plus_b(dropout_out, W, b, name="logits")
      self.scores = tf.nn.softmax(logits)

    with tf.variable_scope('training'):
      self.label = tf.placeholder(tf.int32, shape=[None, 2], name='labels')
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
        labels=self.label, logits=logits))
      self.learning_rate = tf.Variable(FLAGS.init_lr, trainable=False,
        name="learning_rate")
      self.lr_decay_op = self.learning_rate.assign(\
        self.learning_rate * FLAGS.lr_decay)
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
      self.train_op = tf.train.AdamOptimizer(FLAGS.init_lr) \
          .minimize(self.loss, self.global_step)
      self.saver = tf.train.Saver(tf.global_variables())


  def convolution_max_pool_layers(self, embed, \
   sequence_length, filter_sizes, scope):
    embed_expanded = tf.expand_dims(embed, -1)
    pooled_outputs = []
    pooled_size = FLAGS.num_filters * len(filter_sizes) * 2
    for i, filter_size in enumerate(filter_sizes):
      if filter_size > sequence_length:
        pooled_size = FLAGS.num_filters * i * 2
        break
      with tf.variable_scope('%s_%d'%(scope, i)):
        with tf.variable_scope('convolution_layer'):
          filter_shape = [filter_size, FLAGS.embed_size, 1, FLAGS.num_filters]
          W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
          b = tf.get_variable('b', shape=[FLAGS.num_filters], \
            initializer=tf.random_normal_initializer())
          conv = tf.nn.conv2d(
            embed_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
          conv_out = tf.nn.bias_add(conv, b)
        with tf.variable_scope('non-linearity'):
          tanh_out = tf.nn.tanh(conv_out, name="tanh")

        with tf.variable_scope('max_pool'):
          pooled = tf.nn.max_pool(
            tanh_out,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="max_pool")
        pooled_outputs.append(pooled)
    concated_pooled = tf.concat(pooled_outputs, 3)
    pooled_flat = tf.reshape(concated_pooled, [-1, pooled_size])
    return pooled_flat, pooled_size


  def embed_with_overlap(self, sequence, overlap):
    embed=tf.nn.embedding_lookup(self.embed_matrix, sequence)
    expand = tf.expand_dims(overlap, -1)
    return tf.concat([embed, expand], -1)

  def load_word2vec(self, vocab2idx):
    print 'Loading word2vec embedding...'
    with open(FLAGS.word2vec_file) as fin:
      _, dim = map(int, fin.readline().split())
      ret = [0. for _ in xrange(len(vocab2idx)+2)]
      ret[-1] = [0. for _ in xrange(dim)]
      ret[-2] = map(float, fin.readline().rstrip().split()[1:])
      for line in tqdm(fin.readlines()):
        ps = line.rstrip().split()
        if ps[0] in vocab2idx:
          ret[vocab2idx[ps[0]]] = map(float, ps[1:])

    return ret

  def load_vocab(self):
    vocab2idx = {}
    idx = 0
    for line in open(FLAGS.vocab_file):
      ps = line.split(' ')
      if int(ps[1]) < 5:
        break
      vocab2idx[ps[0]] = idx
      idx+=1
    return vocab2idx

  def process_tokens(self, tokens, max_length, vocab_size):
    tokens = map(lambda x: x if x < vocab_size else vocab_size, tokens)
    zero_id = vocab_size+1
    l = len(tokens)
    if l > max_length:
      return tokens[:max_length]
    else:
      tokens.extend(zero_id for _ in xrange(max_length-l))
      return tokens

  def load_data(self, fpath, vocab_size):
    ret=[]
    for line in open(fpath):
      ps = line.split('\t')
      question = map(int, ps[0].split(' '))
      answer = map(int, ps[1].split(' '))
      qo = [0 for _ in xrange(FLAGS.question_max_length)]
      ao = [0 for _ in xrange(FLAGS.answer_max_length)]
      for i, q in enumerate(question):
        if i >= FLAGS.question_max_length:
          break
        for j, a in enumerate(answer):
          if j >= FLAGS.answer_max_length:
            break
          if q==a:
            qo[i]=1
            ao[j]=1
      ret.append([
        self.process_tokens(question, FLAGS.question_max_length, vocab_size), \
        self.process_tokens(answer, FLAGS.answer_max_length, vocab_size), \
        qo, ao,
        [0, 1] if int(ps[2]) == 1 else [1, 0]])
    return ret

  def make_input(self, batch_data):
    return {
      self.question: [data[0] for data in batch_data],
      self.answer: [data[1] for data in batch_data],
      self.question_overlap: [data[2] for data in batch_data],
      self.answer_overlap: [data[3] for data in batch_data],
      self.label: [data[-1] for data in batch_data],
      self.training: not FLAGS.test
    }

  def init_model(self):
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      print ('Restoring model from %s'%ckpt.model_checkpoint_path)
      self.saver.restore(self.session, ckpt.model_checkpoint_path)
    else:
      print ('Initializing model variables')
      self.session.run(tf.global_variables_initializer())

  def train(self):
    data = self.load_data(FLAGS.train_file, self.vocab_size)
    data_size = len(data)
    indices = np.arange(data_size)

    output_feed = [self.global_step, self.train_op, self.loss]
    checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")

    loss = 0.0
    prev_loss=[float('inf')]
    with self.session.as_default():
      self.init_model()

      print 'Start training...'
      while True:
        shuffle_indices = np.random.permutation(indices)
        for i in xrange(0, data_size, FLAGS.batch_size):
          upper = min(i+FLAGS.batch_size, data_size)
          step, _, l = self.session.run(output_feed, \
            self.make_input(data[i:upper]))
          loss += l
          if step % FLAGS.ckpt_per_steps == 0:
            loss /= FLAGS.ckpt_per_steps
            print ("global_step %d, cross entropy %f, learning rate %f"%(
              step, loss, self.learning_rate.eval()))
            sys.stdout.flush()

            if loss > max(prev_loss):
              self.session.run(self.lr_decay_op)
            prev_loss = (prev_loss+[l])[-5:]
            loss = 0.

            self.saver.save(self.session, checkpoint_path, \
              global_step=self.global_step)

  def test(self):
    data = self.load_data(FLAGS.test_file, self.vocab_size)
    data_size = len(data)

    print 'Start testing...'
    with self.session.as_default(), open(FLAGS.result_file, 'w') as fout:
      self.init_model()
      for i in tqdm(xrange(0, data_size, FLAGS.batch_size)):
        upper = min(i+FLAGS.batch_size, data_size)
        batch_data=data[i:upper]
        scores = self.session.run(self.scores, \
          self.make_input(batch_data))
        for s in scores:
          y=np.argmax(s)
          fout.write('%f\n'%(s[y]+y))

def main(_):
  if FLAGS.test:
    DBQA().test()
  else:
    DBQA().train()


if __name__ == '__main__':
tf.app.run()