

import tensorflow as tf

class Model(object):

    def __init__(self,embeddings):

        self.embeddings = embeddings

    def model_fn(self,features,labels,mode,params):

        self.question = features['s1_id']
        self.answer = features['s2_id']
        self.y = labels
        self.num_classes = params['num_classes']
        self.embedding_size = params['embedding_size']
        self.trainable = params['trainable']
        self.optim_type = params['optim_type']
        self.learning_rate = params['learning_rate']
        self.vocab_size = params['vocab_size']

        self.mode = mode

        self._build_graph()

        predictions = {"prob":self.predictions,'score':self.scores[:,1]}

        # export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}

        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode = self.mode,
                predictions = predictions)

        accuracy = tf.metrics.accuracy(labels, self.predictions,name = 'acc_op')
        tf.summary.scalar('accuracy',accuracy[1])

        metrics = {'accuracy':accuracy}
   
        self._create_loss()
        self._create_op()
        if self.mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode = self.mode,
                loss = self.loss)

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode = self.mode,
                loss = self.loss,
                train_op = self.train_op)

    def _build_graph(self):

        self._add_embedding()
        self._feed_neural_network()
        
    def _add_embedding(self):
        with tf.device('/cpu:0'),tf.variable_scope("word_embedding"):

            if self.embeddings is not None:
                self.word_embeddings = tf.Variable(self.embeddings,trainable = self.trainable)
            else:

                self.word_embeddings = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0),names = 'embeddings')

            self.q_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.word_embeddings,self.question),1,name = 'sum_embedding_1')
            self.a_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.word_embeddings,self.answer),1,name = 'sum_embedding_2')

    def _feed_neural_network(self):
        with tf.name_scope('neural_network'):
            self.feature = tf.concat(
              [self.q_emb, self.a_emb], 1, name = 'feature')

            first_hidden_layer = tf.contrib.layers.fully_connected(self.feature,100, activation_fn = tf.nn.relu)
            second_hidden_layer = tf.contrib.layers.fully_connected(first_hidden_layer, 10, activation_fn=tf.nn.relu)
            self.logits = tf.contrib.layers.fully_connected(second_hidden_layer,2,activation_fn = None)
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.logits, 1, name='predictions') 

    def _create_loss(self):
        with tf.name_scope('loss'):

            self.one_hot_labels = tf.one_hot(self.y, self.num_classes)

            losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits = self.logits, labels = self.one_hot_labels))

            correct_prediction = tf.equal(
                      tf.cast(self.predictions, tf.int32), tf.cast(self.y, tf.int32))
            self.accuracy = tf.reduce_mean(
                      tf.cast(correct_prediction, tf.float32), name='Accuracy')

            self.loss = tf.reduce_mean(losses)
            tf.summary.scalar("loss", losses)

    def _create_op(self):

        if self.optim_type == 'adagrad':
          self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
          self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rorop':
          self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
          self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
          raise NotImplementedError(
              'unsupported optimizer:{}'.format(self.optim_type))
        
        self.train_op = self.optimizer.minimize(self.loss, global_step = tf.train.get_global_step())
    
