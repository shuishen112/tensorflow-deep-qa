
import tensorflow as tf
import sys
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('..')

############### CMD Arguments #####################

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("embedding_size",100,"embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("num_classes", 2, "Number of classes")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_string("optim_type", 'adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_boolean("trainable",False," whether train my embedding")
tf.app.flags.DEFINE_integer("num_threads",8,"number of threads")
tf.app.flags.DEFINE_integer("log_steps", 100, "save summary every steps")
tf.app.flags.DEFINE_string("model_type","cnn","model type")
####### dir
tf.app.flags.DEFINE_string("model_dir", 'data/model/', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("data_path",'data/trec',"data set")
tf.app.flags.DEFINE_string("train_tf_records",'data/trec/train.tfrecords','train_tf_records')
tf.app.flags.DEFINE_string("test_tf_records",'data/trec/test.tfrecords','test_tf_records')
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("vocab_dir",'data/trec/vocab','vocab_dir of dataset')
tf.app.flags.DEFINE_string("log_path",None,'log dir of dataset')
tf.app.flags.DEFINE_string("embedding_dir",None,"embedding dir")
# --task_type=train --model_dir= 'data/ceph/zhansu/data/model' --data_path = 'data/ceph/zhansu/data/trec'
# --train_tf_records='data/ceph/zhansu/data/train_tf_records' --test_tf_records='data/ceph/zhansu/data/test_tf_records'
# --vocab_dir="data/ceph/zhansu/data/trec/vocab" --embedding_dir='data/ceph/zhansu/embedding/glove.6B.100d.txt'

tf.app.flags.DEFINE_boolean("debug", False,"use a part of the dataset")
tf.app.flags.DEFINE_boolean("clean",False,"use the clean version")
tf.app.flags.DEFINE_string("task_type", 'prepare', "task type {prepare,train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model",True, "clear existing model or not")
