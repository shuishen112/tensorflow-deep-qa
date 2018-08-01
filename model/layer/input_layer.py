

import tensorflow as tf
import numpy as np
def pair_wise_input():
    q = tf.placeholder(tf.int32, [None,None] ,name = 'input_q')
    a = tf.placeholder(tf.int32, [None] ,name = 'input_app')
    a_neg = tf.placeholder(tf.int32, [None], name = 'input_neg_app')

    dropout_keep_prob = tf.placeholder(
    tf.float32, name='dropout_keep_prob')

    return (q,a,a_neg,dropout_keep_prob)

def point_wise_input():

    q = tf.placeholder(tf.int32,[None,None] ,name = 'input_q')
    a = tf.placeholder(tf.int32,[None],name = 'input_app')
    y = tf.placeholder(tf.int32, [None],name = "input_y")

    dropout_keep_prob = tf.placeholder(
        tf.float32, name='dropout_keep_prob')

    return (q,a,y,dropout_keep_prob)

def embedding_layer(query_embedding,query_embedding_size,app_embedding,app_embedding_size):
    with tf.device('/cpu:0'), tf.variable_scope("word_embedding"):
        # one_hot embedding or pretrained embedding
        if query_embedding is not None:
            query_embed = tf.get_variable(
                'query_embeddings',
                dtype = tf.float32,
                shape=(len(query_embedding), query_embedding_size),
                initializer=tf.constant_initializer(query_embedding),
                trainable= True)
        else:
            print("some problem")

        if app_embedding is not None:
            app_embed = tf.get_variable(
                "app_embedding",
                dtype = tf.float32,
                shape = (len(app_embedding),app_embedding_size),
                initializer = tf.constant_initializer(app_embedding),
                trainable = False) # notive that the app embedding is not be trained
        return query_embedding,app_embedding

def fully_connected_layer(inputs, in_size, out_size, activation_function = None, l = 'layer1'):
    wlimit = np.sqrt(6.0 / (in_size + out_size))
    
    Weights = tf.Variable(tf.random_uniform([in_size, out_size], -wlimit, wlimit),name = l + '_weights')
    biases = tf.Variable(tf.random_uniform([out_size],-wlimit, wlimit),name = l + 'b')
    Wx_plus_b = tf.matmul(tf.cast(inputs,tf.float32), Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def convolution_encode(query,query_length,query_embedding_size,filter_sizes,num_filters):
    with tf.name_scope('convolution_encode'):
        q_emb = tf.expand_dims(query, -1, name='q_emb')
        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size,
                              query_embedding_size, 1, num_filters]
                W = tf.get_variable('Wc' + str(i), filter_shape, tf.float32,
                                  tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
                b = tf.get_variable(
                  'bc' + str(i), [num_filters], tf.float32, tf.constant_initializer(0.01))

                conv = tf.nn.conv2d(
                    tf.cast(q_emb,tf.float32),
                    W,
                    strides = [1,1,query_embedding_size,1],
                    padding = 'SAME',
                    name = 'conv-{}'.format(i)
                    )

                h = tf.nn.relu(tf.nn.bias_add(conv,b),name = 'relu-{}'.format(i))

                pooled = tf.nn.max_pool(
                    h,
                    ksize = [1, query_length,1,1],
                    strides = [1,1,1,1],
                    padding = 'VALID',
                    name = 'pool')

                outputs.append(pooled)

        # outputs concat
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(outputs,3)
        query_encode = tf.reshape(h_pool,[-1,num_filters_total])

