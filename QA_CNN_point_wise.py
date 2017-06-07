#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
# point_wise obbject
class QA(object):
    def __init__(
      self, max_input_left, max_input_right, vocab_size,embedding_size,batch_size,
      embeddings,dropout_keep_prob,filter_sizes, 
      num_filters,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,is_overlap = False,pooling = 'max'):

        self.question = tf.placeholder(tf.int32,[None,max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_answer')
        self.input_y = tf.placeholder(tf.float32, [None,2], name = "input_y")
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.l2_reg_lambda = l2_reg_lambda
        self.filter_sizes = filter_sizes
        self.para = []
        self.hidden_num = 10
        # Embedding layer for both CNN
        with tf.name_scope("embedding"):
            if is_Embedding_Needed:
                print "load embedding"
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = trainable )
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = trainable)
            self.embedding_W = W
            self.embedded_chars_q = tf.expand_dims(tf.nn.embedding_lookup(W,self.question),-1)
            self.embedded_chars_a = tf.expand_dims(tf.nn.embedding_lookup(W,self.answer),-1)
            for p in [self.embedding_W,self.embedded_chars_q,self.embedded_chars_a]:
                self.para.append(p)
        self.kernels = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size,self.embedding_size,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.01), name = "W")
                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name = "b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)

        num_filters_total = num_filters * len(filter_sizes)
        q_conv = self.wide_convolution(self.embedded_chars_q)
        a_conv = self.wide_convolution(self.embedded_chars_a)
        q_pooling = tf.reshape(self.max_pooling(q_conv,max_input_left),[-1,num_filters_total])
        a_pooling = tf.reshape(self.max_pooling(a_conv,max_input_right),[-1,num_filters_total])
        # concat the input vector to classification task
        self.feature = tf.concat(1,[q_pooling,a_pooling],name = 'feature')

        with tf.name_scope('neural_network'):
            W = tf.get_variable(
                "W_hidden",
                shape=[2 * num_filters_total, self.hidden_num],
                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.0, shape = [self.hidden_num]), name = "b")
            self.para.append(W)
            self.para.append(b)
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.feature, W, b, name = "hidden_output"))

        #add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape = [self.hidden_num, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.0, shape=[2]), name="b")
            self.para.append(W)
            self.para.append(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
            self.see = self.scores
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def max_pooling(self,conv,input_length):
        pooled = tf.nn.max_pool(
                    conv,
                    ksize = [1, input_length, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name="pool")
        return pooled

    def wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, self.embedding_size, 1],
                    padding='SAME',
                    name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(3,cnn_outputs)
        return cnn_reshaped
    def narrow_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv-1"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name="relu-1")
            cnn_outputs.append(h)
        # cnn_reshaped = tf.concat(3,cnn_outputs)
        return cnn_outputs
        # return cnn_reshaped

if __name__ == '__main__':
    cnn = QA(max_input_left = 33,
                max_input_right = 40,
                vocab_size = 5000,
                embedding_size = 100,
                batch_size = 3,
                embeddings = None,
                dropout_keep_prob = 1,
                filter_sizes = [3,4,5],
                num_filters = 64,
                l2_reg_lambda = 0.0,
                is_Embedding_Needed = False,
                trainable = True,
                is_overlap = False)
  
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_y = np.ones((3,2))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.input_y:input_y
        }
       
        question,answer,see = sess.run([cnn.question,cnn.answer,cnn.see],feed_dict)
        print see

       
