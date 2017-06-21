import tensorflow as tf 
import time

def test_variable_scope():
	with tf.variable_scope("foo"):
	    v = tf.get_variable("v", [1])
	with tf.variable_scope("foo", reuse=True):
	    v1 = tf.get_variable("v", [1])
	assert v1 is v
def test_tf_queue():
	x_input_data = tf.random_normal([3],mean = -1,stddev = 4)

	q = tf.FIFOQueue(capacity = 3,dtypes = tf.float32)

	enqueue_op = q.enqueue_many(x_input_data)

	input = q.dequeue()

	input = tf.Print(input,data = [q.size()],message = 'Nb elemets left:')

	y = input + 1

	with tf.Session() as sess:
		sess.run(enqueue_op)
		sess.run(y)
		sess.run(y)
		sess.run(y)
		# now the queue is empty,if we call it again,our program will hang right

		sess.run(y)
'''
def read_and_decode(filename_queue):
	reader = tf.TextLineReader()
	key,value = reader.read(filename_queue)
	record_defaults = [['id'], ['qid'], ['aid'], ['question1'], ['question2'],['flag']]
	#default values,in case of empty columns
	col1,col2,col3,col4,col5,col6 = tf.decode_csv(value,record_defaults = record_defaults,field_delim = ',')
	features = tf.stack([col4,col5])
	return tf.train.shuffle_batch([features,col6],
		batch_size = 32,
		num_threads = 1,
		capacity = 50000,
		min_after_dequeue = 10000,
		allow_smaller_final_batch = True)
		'''
def test_read():
	# import pandas as pd
	# file = 'data/quora/train.csv'
	# df = pd.read_csv(file)
	# print df

	filename_queue = tf.train.string_input_producer(['data/quora/train.csv'])
	features,label = read_and_decode(filename_queue)
	# The op for initializing the variables.
	init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
	with tf.Session() as sess:
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord = coord)

		# for i in range(1000):
			#retrieve a single instance:

		example = sess.run([label])
		print example

		coord.request_stop()
		coord.join(threads)
def _bytes_feature(value):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def convert_to_tfrecords():
	import pandas as pd
	import numpy as np 
	tfrecords_filename = 'quora.tfrecords'
	filename_pairs = ['data/quora/train.csv']
	writer = tf.python_io.TFRecordWriter(tfrecords_filename)

	for file_name in filename_pairs:
		df = pd.read_csv("data/quora/train.csv").fillna("")
		for row in df:
			question1 = df['question1']
			question2 = df['question2']
			flag = df['is_duplicate']

			example = tf.train.Example(features = tf.train.Features(feature = {
				'question1':_bytes_feature(np.array(df['question1'].tolist()).tostring()),
				'question2':_bytes_feature(np.array(df['question2'].tolist()).tostring())}
				))
			writer.write(example.SerializeToString())
	writer.close()
def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		#defaults are not specified since both keys are required
		features = {
		'question1':tf.FixedLenFeature([],tf.string),
		'question2':tf.FixedLenFeature([],tf.string)
		})
	question1 = tf.decode_raw(features['question1'],tf.uint8)
	question2 = tf.decode_raw(features['question2'],tf.uint8)
	question1 = tf.reshape(question1,[-1,1])
	question2 = tf.reshape(question2,[-1,1])
	print question2
	return tf.train.shuffle_batch([question1,question2],
		batch_size = 20,
		capacity = 30,
		num_threads = 2,
		shapes = 20 ,
		min_after_dequeue = 10)

filename_queue = tf.train.string_input_producer(
	['quora.tfrecords'])
question1,question2 = read_and_decode(filename_queue)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	for i in xrange(3):
		q1,q2 = sess.run([question1,question2])
		print q1,q2
	coord.request_stop()
	coord.join(threads)