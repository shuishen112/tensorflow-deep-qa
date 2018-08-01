

import tensorflow as tf

from model.layer.input_layer import fully_connected_layer

def model_fn(features, labels,mode, params):


    # question = tf.cast(tf.reshape(features['s1_id'],[-1,40]),tf.float32)
    # answer = tf.cast(tf.reshape(features['s2_id'],[-1,40]),tf.float32)

    question = features['s1_id']
    answer = features['s2_id']


    labels = tf.cast(labels,tf.float32)
    # 1. Configure the model via TensorFlow operations

    first_hidden_layer = tf.contrib.layers.fully_connected(tf.concat([question,answer],1),10, activation_fn = tf.nn.relu)
    # first_hidden_layer = tf.layers.dense(tf.concat([question,answer],1),10, activation = tf.nn.relu)

    # second_hidden_layer = tf.layers.dense(
      # first_hidden_layer, 10, activation=tf.nn.relu)
    second_hidden_layer = tf.contrib.layers.fully_connected(first_hidden_layer, 10, activation_fn=tf.nn.relu)

    # output_layer = tf.layers.dense(second_hidden_layer, 1)
    output_layer = tf.contrib.layers.fully_connected(second_hidden_layer,1)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     pred = sess.run(predictions)
    #     y = sess.run(labels)

    #     print(pred)
    #     print(y)

    #     exit()
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"ages": predictions})

    # Calculate loss using mean squared error

    labels = tf.cast(labels,tf.float32)
    predictions = tf.cast(predictions,tf.float32)

    assert labels.dtype == predictions.dtype

    loss = tf.nn.softmax_cross_entropy_with_logits(labels =labels,logits=predictions)

    tf.summary.scalar("loss", loss)
    #loss = tf.losses.mean_squared_error(labels = labels, predictions = predictions)

    print('labels.type:{} --- predictions.type:{}'.format(labels.dtype, predictions.dtype))
    # assert labels.dtype == predictions.dtype
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.float32), predictions)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
    
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
            
            # in_size = self.embedding_size * 2

            # out_size = in_size
            
            # self.layer1_outputs = fully_connected_layer(self.feature, in_size, out_size,tf.nn.relu, l = 'layer1')

            # self.W_projection = tf.get_variable(
            #   'w_hidden',
            #   shape=[out_size, self.num_classes],
            #   initializer=tf.contrib.layers.xavier_initializer()
            # )

            # self.b_projection = tf.get_variable(
            #   'b_hidden',
            #   shape=[self.num_classes]
            # )


            # self.logits = tf.nn.xw_plus_b(self.layer1_outputs, self.W_projection,self.b_projection, name='logits')
            # self.logits = self.add_layer(self.feature,in_size,out_size,None)

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
    
