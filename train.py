#coding=utf-8
#! /usr/bin/env python3.4
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helpers import get_overlap_dict,load,prepare,batch_gen_with_single,batch_gen_with_point_wise
import operator
from QA_CNN_point_wise import QA
import random
import evaluation
import cPickle as pickle
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

now = int(time.time()) 
    
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)

from functools import wraps
#print( tf.__version__)
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco



# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim",50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000001, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learn rate( default: 0.0)")
tf.flags.DEFINE_integer("max_len_left", 40, "max document length of left input")
tf.flags.DEFINE_integer("max_len_right", 40, "max document length of right input")
tf.flags.DEFINE_string("loss","point_wise","loss function (default:point_wise)")
tf.flags.DEFINE_integer('extend_feature_dim',10,'overlap_feature_dim')
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_boolean("trainable", False, "is embedding trainable? (default: False)")
tf.flags.DEFINE_integer("num_epochs", 500, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_boolean('overlap_needed',True,"is overlap used")
tf.flags.DEFINE_boolean('dns','False','whether use dns or not')
tf.flags.DEFINE_string('data','wiki','data set')
tf.flags.DEFINE_string('CNN_type','qacnn','data set')
tf.flags.DEFINE_float('sample_train',1,'sampe my train data')
tf.flags.DEFINE_boolean('fresh',True,'wheather recalculate the embedding or overlap default is True')
tf.flags.DEFINE_string('pooling','max','pooling strategy')
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
print FLAGS.__flags
log_dir = 'log/'+ timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
precision = data_file + 'precise'

@log_time_delta
def predict(sess,cnn,test,alphabet,batch_size,q_len,a_len):
    scores = []
    d = get_overlap_dict(test,alphabet,q_len,a_len)
    for data in batch_gen_with_single(test,alphabet,batch_size,q_len,a_len,overlap_dict = d): 
        feed_dict = {
            cnn.question: data[0],
            cnn.answer: data[1],
            cnn.q_overlap:data[2],
            cnn.a_overlap:data[3]
        }
        score = sess.run(cnn.scores,feed_dict)
        scores.extend(score)
    return np.array(scores[:len(test)])

@log_time_delta
def test_point_wise():
    train,test,dev = load(FLAGS.data,filter = True)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print 'train question unique:{}'.format(len(train['question'].unique()))
    print 'train length',len(train)
    print 'test length', len(test)
    print 'dev length', len(dev)

    alphabet,embeddings = prepare([train,test,dev],dim = FLAGS.embedding_dim,is_embedding_needed = True,fresh = True)
    print 'alphabet:',len(alphabet)
    with tf.Graph().as_default():
        # with tf.device("/cpu:0"):
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default(),open(precision,"w") as log:

            # train,test,dev = load("trec",filter=True)
            # alphabet,embeddings = prepare([train,test,dev],is_embedding_needed = True)
            cnn = QA(
                max_input_left = q_max_sent_length,
                max_input_right = a_max_sent_length,
                vocab_size = len(alphabet),
                embedding_size = FLAGS.embedding_dim,
                batch_size = FLAGS.batch_size,
                embeddings = embeddings,
                dropout_keep_prob = FLAGS.dropout_keep_prob,
                filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters = FLAGS.num_filters,
                l2_reg_lambda = FLAGS.l2_reg_lambda,
                is_Embedding_Needed = True,
                trainable = FLAGS.trainable,
                overlap_needed = FLAGS.overlap_needed,
                pooling = FLAGS.pooling)

            # Define Training procedure
            global_step = tf.Variable(0, name = "global_step", trainable = False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # seq_process(train, alphabet)
            # seq_process(test, alphabet)
            map_max = 0.65
            for i in range(100):
                d = get_overlap_dict(train,alphabet,q_len = q_max_sent_length,a_len = a_max_sent_length)
                datas = batch_gen_with_point_wise(train,alphabet,FLAGS.batch_size,overlap_dict = d,
                    q_len = q_max_sent_length,a_len = a_max_sent_length)
                for data in datas:
                    feed_dict = {
                        cnn.question:data[0],
                        cnn.answer:data[1],
                        cnn.input_y:data[2],
                        cnn.q_overlap:data[3],
                        cnn.a_overlap:data[4]
                    }
                    _, step,loss, accuracy,pred ,scores,see = sess.run(
                    [train_op, global_step,cnn.loss, cnn.accuracy,cnn.predictions,cnn.scores,cnn.see],
                    feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}  ".format(time_str, step, loss, accuracy))
                
                    # print loss
                predicted_train = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                map_mrr_train = evaluation.evaluationBypandas(train,predicted[:,-1])
                predicted = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                map_mrr_test = evaluation.evaluationBypandas(test,predicted[:,-1])

                if map_mrr_test[0] > map_max:
                        map_max = map_mrr_test[0]
                        timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
                        folder = 'runs/' + timeDay
                        out_dir = folder +'/'+timeStamp+'__'+FLAGS.data+str(map_mrr_test[0])
                        if not os.path.exists(folder):
                            os.makedirs(folder)
                        save_path = saver.save(sess, out_dir)
                        print "Model saved in file: ", save_path
                # predicted = predict(sess,cnn,dev,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length)
                # map_mrr_dev = evaluation.evaluationBypandas(dev,predicted[:,-1])
                # map_mrr_train = evaluation.evaluationBypandas(train,predicted_train[:,-1])
                # print evaluation.evaluationBypandas(train,predicted_train[:,-1])
                print "{}:train epoch:map mrr {}".format(i,map_mrr_train)
                print "{}:test epoch:map mrr {}".format(i,map_mrr_test)
                # print "{}:dev epoch:map mrr {}".format(i,map_mrr_dev)
                # line = " {}:epoch: map_train{}----map_test{}----map_dev{}".format(i,map_mrr_train[0],map_mrr_test[0],map_mrr_dev[0])
                line = " {}:epoch: map_test{}".format(i,map_mrr_test[0])
                log.write(line + '\n')
                log.flush()
            log.close()
if __name__ == '__main__':
    # test_quora()
    if FLAGS.loss == 'point_wise':
        test_point_wise()
    # test_pair_wise()
    # test_point_wise()
