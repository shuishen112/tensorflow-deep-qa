#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
# point_wise obbject
class QA(object):
    def __init__(
      self, max_input_left, max_input_right, vocab_size,embedding_size,batch_size,
      embeddings,dropout_keep_prob,filter_sizes, 
      num_filters,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,overlap_needed = True,pooling = 'max',hidden_num = 10,\
      extend_feature_dim = 10):

        self.question = tf.placeholder(tf.int32,[None,max_input_left],name = 'input_question')
        self.answer = tf.placeholder(tf.int32,[None,max_input_right],name = 'input_answer')
        self.input_y = tf.placeholder(tf.float32, [None,2], name = "input_y")
        self.dropout_keep_prob = dropout_keep_prob
        self.num_filters = num_filters
        self.embeddings = embeddings
        self.overlap_needed = overlap_needed
        if self.overlap_needed:
            self.total_embedding_dim = embedding_size + extend_feature_dim
        else:
            self.total_embedding_dim = embedding_size
        self.batch_size = batch_size
        self.l2_reg_lambda = l2_reg_lambda
        self.filter_sizes = filter_sizes
        self.para = []
        self.max_input_left = max_input_left
        self.max_input_right = max_input_right
        self.hidden_num = hidden_num
        self.extend_feature_dim = extend_feature_dim

        self.q_overlap = tf.placeholder(tf.int32,[None,max_input_left],name = 'q_feature_embeding')
        self.a_overlap = tf.placeholder(tf.int32,[None,max_input_right],name = 'a_feature_embeding')
        # Embedding layer for both CNN
        with tf.name_scope("embedding"):
            if is_Embedding_Needed:
                print "load embedding"
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = trainable )
            else:
                W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W",trainable = trainable)
            self.embedding_W = W
            self.overlap_W = tf.Variable(tf.random_uniform([3, self.extend_feature_dim], -1.0, 1.0),name="W",trainable = True)
            self.para.append(self.embedding_W)
            self.para.append(self.overlap_W)

        #get embedding from the word indices
        self.embedded_chars_q = self.getEmbedding(self.question,self.q_overlap)
        print self.embedded_chars_q
        self.embedded_chars_a = self.getEmbedding(self.answer,self.a_overlap)

        #initialize my conv kernel
        self.kernels = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size,self.total_embedding_dim,1,num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.01), name = "W")
                b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name = "b")
                self.kernels.append((W,b))
                self.para.append(W)
                self.para.append(b)

        num_filters_total = num_filters * len(filter_sizes)
        q_conv = self.wide_convolution(self.embedded_chars_q)
        a_conv = self.wide_convolution(self.embedded_chars_a)
        with tf.name_scope('pooling'):
            #different pooling strategy. max pooing or attentive pooling
            if pooling == 'max':
                print pooling
                self.q_pooling = tf.reshape(self.max_pooling(q_conv,max_input_left),[-1,num_filters_total])
                self.a_pooling = tf.reshape(self.max_pooling(a_conv,max_input_right),[-1,num_filters_total])
            elif pooling == 'attentive':
                print pooling
                with tf.name_scope('attention'):    
                    self.U = tf.Variable(tf.truncated_normal(shape = [num_filters_total,num_filters_total],stddev = 0.01,name = 'U'))
                    self.para.append(self.U)

                self.q_pooling,self.a_pooling = self.attentive_pooling(q_conv,a_conv)
                self.q_pooling = tf.reshape(self.q_pooling,[-1,num_filters_total])
                self.a_pooling = tf.reshape(self.a_pooling,[-1,num_filters_total])
            else:
                print 'no pooling'
            
        # Compute similarity
        with tf.name_scope("similarity"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer())
            self.transform_left = tf.matmul(self.q_pooling, W)
            self.sims = tf.reduce_sum(tf.mul(self.transform_left, self.a_pooling), 1, keep_dims=True)
            self.para.append(W)
            print W
            self.see = W
        # concat the input vector to classification task
        self.feature = tf.concat(1,[self.q_pooling,self.sims,self.a_pooling],name = 'feature')

        with tf.name_scope('neural_network'):
            W = tf.get_variable(
                "W_hidden",
                shape=[2 * num_filters_total + 1, self.hidden_num],
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
                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.0, shape=[2]), name="b")
            self.para.append(W)
            self.para.append(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name = "scores")
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
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


    def getEmbedding(self,words_indice,overlap_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        if not self.overlap_needed:
            print 'hello world'
            return  tf.expand_dims(embedded_chars_q,-1)

        overlap_embedding_q = tf.nn.embedding_lookup(self.overlap_W,overlap_indice)
        return  tf.expand_dims(tf.concat(2,[embedded_chars_q,overlap_embedding_q]),-1)

    def max_pooling(self,conv,input_length):
        pooled = tf.nn.max_pool(
                    conv,
                    ksize = [1, input_length, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name="pool")
        return pooled
    def attentive_pooling(self,input_left,input_right):
        Q = tf.reshape(input_left,[self.batch_size,self.max_input_left,len(self.filter_sizes) * self.num_filters],name = 'Q')
        A = tf.reshape(input_right,[self.batch_size,self.max_input_right,len(self.filter_sizes) * self.num_filters],name = 'A')

        # G = tf.tanh(tf.matmul(tf.matmul(Q,self.U),\
        # A,transpose_b = True),name = 'G')
        first = tf.matmul(tf.reshape(Q,[-1,len(self.filter_sizes) * self.num_filters]),self.U)
        second_step = tf.reshape(first,[self.batch_size,-1,len(self.filter_sizes) * self.num_filters])
        result = tf.batch_matmul(second_step,tf.transpose(A,perm = [0,2,1]))
        G = tf.tanh(result)
        # column-wise pooling ,row-wise pooling
        row_pooling = tf.reduce_max(G,1,True,name = 'row_pooling')
        col_pooling = tf.reduce_max(G,2,True,name = 'col_pooling')

        attention_q = tf.nn.softmax(col_pooling,1,name = 'attention_q')
        attention_a = tf.nn.softmax(row_pooling,name = 'attention_a')

        R_q = tf.reshape(tf.matmul(Q,attention_q,transpose_a = 1),[self.batch_size,self.num_filters * len(self.filter_sizes),-1],name = 'R_q')
        R_a = tf.reshape(tf.matmul(attention_a,A),[self.batch_size,self.num_filters * len(self.filter_sizes),-1],name = 'R_a')

        return R_q,R_a
        
    def wide_convolution(self,embedding):
        cnn_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                    embedding,
                    self.kernels[i][0],
                    strides=[1, 1, self.total_embedding_dim, 1],
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
                overlap_needed = True)
  
    input_x_1 = np.reshape(np.arange(3 * 33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_y = np.ones((3,2))

    input_overlap_q = np.ones((3,33))
    input_overlap_a = np.ones((3,40))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.answer:input_x_2,
            cnn.input_y:input_y,
            cnn.q_overlap:input_overlap_q,
            cnn.a_overlap:input_overlap_a
        }
       
        question,answer,scores = sess.run([cnn.question,cnn.answer,cnn.scores],feed_dict)
        print scores

       
