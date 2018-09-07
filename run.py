
from config import FLAGS
from dataset import QA_dataset
import os
from datetime import date,timedelta
import tensorflow as tf
from model.QA_CNN import model_fn
from model.QA_CNN import cnn_model_fn
from model.QA_CNN import cnn_quantum_fn
import evaluation
import pickle
import logging
import shutil
import numpy as np

if FLAGS.model_type == "fnn":

    model_params = {
        "num_classes":FLAGS.num_classes,
        "embedding_size":FLAGS.embedding_size,
        "learning_rate":FLAGS.learning_rate,
        "trainable":FLAGS.trainable,
        "optim_type":FLAGS.optim_type
    }
elif FLAGS.model_type == 'cnn':

    model_params = {
        'query_length' : 40,
        'app_name_length': 40,
        'trainable': False,
        'filter_sizes': [3,4,5],
        'num_filters':64,
        'optim_type':'adam',
        'embedding_size':FLAGS.embedding_size,
        'learning_rate':0.001,
        'batch_size':64,
        "trainable":FLAGS.trainable,
        "num_classes":FLAGS.num_classes
    }
else:
    pass
    
def prepare():

    logger = logging.getLogger('QA')
    data_path = FLAGS.data_path
    train_file = os.path.join(data_path,'train.txt')
    test_file = os.path.join(data_path,'test.txt')
    dev_file = os.path.join(data_path,'test.txt')

    logger.info('checking the data file')
    for dir_path in [FLAGS.vocab_dir,FLAGS.model_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    data_set = QA_dataset(train_file,dev_file,test_file,FLAGS)
    data_set.get_alphabet([data_set.train_set,data_set.test_set])

    # with open ('index_to_word','w',encoding = 'utf-8') as fout:
    #     for index in data_set.index_to_word:
            
    #         line = str(index) + '\t' + data_set.index_to_word[index] + '\n'

    #         fout.write(line)

    

    data_set.process_pairs()

    embeddings = data_set.get_embedding(FLAGS.embedding_dir,data_set.word_dict,dim = FLAGS.embedding_size)

    print("alphabet size{}:".format(len(data_set.word_dict)))

    para = {'embeddings':embeddings}
    logger.info('save the embedding')


    with open(os.path.join(FLAGS.vocab_dir,'vocab.data'),'wb') as fout:
        pickle.dump(para, fout)

    logger.info('Done with preparing')




def train():


    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
        FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    if FLAGS.clear_existing_model:
        print("clear the exist model")
        if os.path.exists(FLAGS.model_dir):
            shutil.rmtree(FLAGS.model_dir,ignore_errors=True)

      
    logger = logging.getLogger('QA')
    logger.info("load vocab")

    with open(os.path.join(FLAGS.vocab_dir,'vocab.data'),'rb') as fin:
        vocab = pickle.load(fin)
    logger.info('loading the dataset')

    data_set = QA_dataset(None,None,None,FLAGS)


    model_params["vocab_size"] = len(vocab['embeddings'])
    model_params["embeddings"] = vocab["embeddings"]
    

    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
                log_step_count_steps=FLAGS.log_steps, save_summary_steps = FLAGS.log_steps)

    QA_CNN = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir = FLAGS.model_dir, params = model_params, config=config)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: data_set.input_fn(FLAGS.train_tf_records, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size,perform_shuffle = True),max_steps = 20000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: data_set.input_fn(FLAGS.test_tf_records, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
    tf.estimator.train_and_evaluate(QA_CNN, train_spec, eval_spec)



def predict():

    logger = logging.getLogger('QA')
    logger.info('load vocab')

    data_path = FLAGS.data_path
    train_file = os.path.join(data_path,'train.txt')
    test_file = os.path.join(data_path,'test.txt')

    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
        FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
    with open(os.path.join(FLAGS.vocab_dir,'vocab.data'),'rb') as fin:
        vocab = pickle.load(fin)

    model_params["vocab_size"] = len(vocab['embeddings'])
    model_params["embeddings"] = vocab["embeddings"]
    data_set = QA_dataset(None,None,test_file,FLAGS) 

    
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
                log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    QA_CNN = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    preds = QA_CNN.predict(input_fn=lambda: data_set.input_fn(FLAGS.test_tf_records, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys=["prob",'score'])
    # list_pred = list(map(lambda x:x['prob'],preds))
    a = list(map(lambda x:(x['prob'],x['score']),preds))
    list_pred, score = zip(*a)

    random_pred = np.random.rand(len(data_set.test_set))
    print('random:{}\n'.format(evaluation.evaluationBypandas(data_set.test_set,random_pred)))

    print(evaluation.evaluationBypandas(data_set.test_set,score))

    # data_set.test_set['pred'] = list_pred
    print(data_set.test_set.head())
    data_set.test_set.to_csv('pred.txt',sep = '\t',index = None,header = None)
   
def main(_):
    

    logger = logging.getLogger('QA')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    
    if FLAGS.log_path:
        file_handler = logging.FileHandler(FLAGS.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


    # os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    if FLAGS.task_type == 'prepare':
        prepare()
    elif FLAGS.task_type == 'train':
        train()
    elif FLAGS.task_type == 'infer':
        predict()
        
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()