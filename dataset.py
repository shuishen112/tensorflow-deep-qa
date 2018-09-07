import pandas as pd
import numpy as np
from collections import Counter
import os
import logging
from tqdm import tqdm, tqdm_pandas
import jieba
import pickle
from functools import wraps
import time
import spacy
import tensorflow as tf
from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')
tqdm.pandas(tqdm,leave = True)

nlp = spacy.blank("en")

def removeUnanswerdQuestion(df):
    counter= df.groupby("s1").apply(lambda group: sum(group["flag"]))
    questions_have_correct=counter[counter>0].index
    counter= df.groupby("s1").apply(lambda group: sum(group["flag"]==0))
    questions_have_uncorrect=counter[counter>0].index
    counter=df.groupby("s1").apply(lambda group: len(group["flag"]))
    questions_multi=counter[counter>1].index

    return df[df["s1"].isin(questions_have_correct) &  df["s1"].isin(questions_have_correct) & df["s1"].isin(questions_have_uncorrect)].reset_index()
# calculate the time
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

def cut(sent):

    words = sent.lower().split()
    words = [w for w in words if w not in en_stopwords]
    return words
    # return sent.lower().split()
def word_overlap(row):
    question = cut(row["s1"]) 
    answer = cut(row["s2"])
    overlap = set(answer).intersection(set(question))
    return len(overlap)
class QA_dataset(object):

    def __init__(self,train_file = None,dev_file = None,test_file = None,args = None):

        self.train_set,self.dev_set,self.test_set = None,None,None
        self.logger = logging.getLogger('QA')
        self.args = args

        if train_file:
            self.train_set = self.load_data(train_file)
            self.logger.info('Train set size: {}'.format(len(self.train_set)))
            print('Train set size: {}'.format(len(self.train_set)))
            print('Train set unique s1:{}'.format(len(self.train_set['s1'].unique())))

        if dev_file:

            self.dev_set = self.load_data(dev_file)
            self.logger.info('dev set size:{}'.format(len(self.dev_set)))
            print('dev set size:{}'.format(len(self.dev_set)))

        if test_file:
            self.test_set = self.load_data(test_file)
            self.logger.info('test set size:{}'.format(len(self.test_set)))
            print('test set size:{}'.format(len(self.test_set)))

    def process_pairs(self):

        # self.df_neg = self.train_set[self.train_set['flag'] == 0]['s2'].reset_index()
        # self.df_train_pairs = self.train_set.groupby('search_id').progress_apply(self.triple_pair).dropna()
        # self.df_test_pairs = self.test_set.groupby('search_id').progress_apply(self.triple_pair).dropna()

        self.df_train_pairs = self.train_set.progress_apply(self.point_wise_pair,axis = 1)
        self.df_test_pairs = self.test_set.progress_apply(self.point_wise_pair,axis = 1)

        tfrecords_filename_train = self.args.train_tf_records
        tfrecords_filename_test = self.args.test_tf_records

        self.build_feature(self.df_train_pairs, tfrecords_filename_train)
        self.build_feature(self.df_test_pairs, tfrecords_filename_test)
    def build_feature(self,df,tf_file_name):

        writer = tf.python_io.TFRecordWriter(tf_file_name)
        for index,row in df.iterrows():
            s1_id = np.array(row['s1_id']).tostring()
            s2_id = np.array(row['s2_id']).tostring()
            flag = row['flag']

            overlap = word_overlap(row)
            example = tf.train.Example(features = tf.train.Features(
                feature = {
                's1_id':tf.train.Feature(bytes_list = tf.train.BytesList(value = [s1_id])),
                's2_id':tf.train.Feature(bytes_list = tf.train.BytesList(value = [s2_id])),
                'flag':tf.train.Feature(int64_list = tf.train.Int64List(value = [flag])),
                'overlap':tf.train.Feature(int64_list = tf.train.Int64List(value = [overlap]))
                }))

            writer.write(example.SerializeToString())
        writer.close()

    def load_data(self,data_path):
        data = pd.read_csv(data_path,sep = '\t',names = ['s1','s2','flag'],quoting = 3)
        if self.args.debug:
            data = data[:1000]
        if self.args.clean:
            data = removeUnanswerdQuestion(data)
        return data

    @log_time_delta
    def get_alphabet(self,corpuses):
        word_counter = Counter()
        for corpus in corpuses:
            for texts in [corpus['s1'].unique(), corpus['s2']]:
                for sentence in texts:
                    tokens = cut(sentence)
                    for token in set(tokens):
                        word_counter[token] += 1
        word_dict = {w: index + 2 for (index, w) in enumerate(list(word_counter))}

        word_dict['NULL'] = 0
        word_dict['UNK'] = 1

        index_to_word = {word_dict[w]: w for w in word_dict}
        self.index_to_word = index_to_word
        self.word_dict = word_dict

        print('alphabet_size: {}'.format(len(self.word_dict)))
        return word_dict
        # print(self.query_dict)
    def get_embedding(self,fname,vocab,dim = 100):
        embeddings = np.random.normal(0,1,size = [len(vocab),dim])

        word_vecs = {}
        count = 0
        with open(fname,encoding = 'utf-8') as f:
            i = 0
            for line in f:
                i += 1
                if i % 100000 == 0:
                    print ('epch %d' % i)
                items = line.strip().split(' ')
                if len(items) == 2:
                  vocab_size, embedding_size = items[0], items[1]
                  print (vocab_size, embedding_size)
                else:
                  word = items[0]
                  if word in vocab:
                    count += 1
                    embeddings[vocab[word]] = items[1:]
        print('there are {} words can be found in dict'.format(count))
        return embeddings

    def convert_to_word_ids(self,sentence,max_len = 40):
        indices = []
        tokens = cut(sentence)
        for word in tokens:
            if word in self.word_dict:
                indices.append(self.word_dict[word])
            else:
                continue
        result = indices + [self.word_dict['NULL']] * (max_len - len(indices))

        return result[:max_len]


    def point_wise_pair(self,row):
        return pd.Series({'s1':row['s1'],'s2':row['s2'],'s1_id':self.convert_to_word_ids(row['s1']),'s2_id':self.convert_to_word_ids(row['s2']),'flag':row['flag']})

    # noting that the code here is different from the previous code
    def triple_pair(self,group):
        question = group['s1'].tolist()
        pos_answer = group[group['flag'] == 1]['s2']
        neg_answer = group[group['flag'] == 0]['s2'].reset_index()

        if len(pos_answer) > 0:
            for pos in pos_answer:
                neg_index = np.random.choice(neg_answer)
                neg = neg_answer.loc[neg_index]['s2']

                return pd.Series({'s1_id':self.convert_to_word_ids(question[0]),
                    's2_pos_id':self.convert_to_word_ids(pos),
                    's2_neg_id':self.convert_to_word_ids(neg)})

    @log_time_delta
    def batch_iter_pandas(self,df,batch_size,shuffle = False,args = None):

        if shuffle:
            df.sample(frac = 1).reset_index(drop = True)

        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0,len(seq),size))

        batches = chunker(df,batch_size)
        for b in batches:
            yield(b['s1_id'].tolist(),b['s2_id'].tolist(),b['flag'].tolist())

    def get_record_parser(self,serialized_example):
        features = tf.parse_single_example(serialized_example,
            features = {
            's1_id': tf.FixedLenFeature([],tf.string),
            's2_id':tf.FixedLenFeature([],tf.string),
            'flag':tf.FixedLenFeature([],tf.int64),
            'overlap':tf.FixedLenFeature([],tf.int64)
            })

        s1_id = tf.decode_raw(features['s1_id'],tf.int32)
        s2_id = tf.decode_raw(features['s2_id'],tf.int32)
        flag = features['flag']
        overlap = features['overlap']
        return {'s1_id':s1_id,'s2_id':s2_id,'overlap':overlap},flag

    def input_fn(self,filenames, batch_size = 32, num_epochs = 1,perform_shuffle = False):
        data_set = tf.data.TFRecordDataset(filenames).map(self.get_record_parser,num_parallel_calls = cpu_count())
        if perform_shuffle:
            data_set = data_set.shuffle(buffer_size=256)

        data_set = data_set.repeat(num_epochs)
        data_set = data_set.batch(batch_size)

        iterator = data_set.make_one_shot_iterator()

        batch_features, batch_labels = iterator.get_next()

        return batch_features, batch_labels

    

# data_path = 'data/trec'

# train_file = os.path.join(data_path,'train.txt')
# test_file = os.path.join(data_path,'test.txt')
# dev_file = os.path.join(data_path,'test.txt')

# class config(object):
#   debug = True
#   loss = 'pair_wise_loss'
#   train_tf_records = 'data/trec/train.tfrecords'
#   test_tf_records = 'data/trec/test.tfrecords'

# args = config()
# data_set = QA_dataset(train_file,dev_file,test_file,args)
# data_set.get_alphabet([data_set.train_set,data_set.test_set])

# # print(data_set.word_dict)
# data_set.process_pairs()
# batch = data_set.batch_iter_pandas(data_set.df_train_pairs,60,shuffle = True,args = args)
# print(data_set.df_train_pairs)

# filenames = 'data/trec/train.tfrecords'
# ds = tf.data.TFRecordDataset(filenames).map(data_set.get_record_parser,num_parallel_calls = 8).prefetch(500000)


# # iterator = ds.make_one_shot_iterator()

# # next_element = iterator.get_next()

# next_element = data_set.input_fn(filenames)

# with tf.Session() as sess:
#     print(sess.run(next_element))

# with tf.Session() as sess:

#     for serialized_example in tf.python_io.tf_record_iterator(filenames):

#         features = tf.parse_single_example(serialized_example,
#             features = {
#             's1_id': tf.FixedLenFeature([],tf.string),
#             's1_id':tf.FixedLenFeature([],tf.string),
#             'flag':tf.FixedLenFeature([],tf.int64)
#             })


#         s1_id = tf.decode_raw(features['s1_id'],tf.int32)


#         print(sess.run(s1_id))

# ds = ds.batch(32)

# iterator = ds.make_one_shot_iterator()

# batch_str = iterator.get_next()





# for d in batch:
#   q,a,a_n = d
#   print(a)
# for d in batch:
#   q,a,a_n = zip(*d)
    # print(q)

# print('positive rate:{}'.format(df['flag'].sum() / len(df)))
# print('positive unique:{}'.format(len(df['query'].unique())))
# print('s2 unique:{}'.format(len(df['s2'].unique())))
# print('number of positive query:{}'.format(df['flag'].sum()))

# # print(data_set.train_set['flag'])
# # print(data_set.test_set)
# data_set.get_alphabet([data_set.train_set,data_set.test_set])

# # embeddings = data_set.get_app_embedding('data/app_embedding',data_set.s2_dict,dim = 150)


# # print(data_set.query_dict)
# # print(data_set.s2_dict)
# batch = data_set.batch_iter(data_set.train_set,60,shuffle = True,args = args)