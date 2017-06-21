import tensorflow as tf 
DATA_PATH = 'data/trec/train.txt'

BATCH_SIZE = 64
N_FEATURES = 3
def batch_generator(filenames):
    """ filenames is the list of files you want to read from. 
    In this case, it contains only heart.csv
    """
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=1) # skip the first line in the file
    _, value = reader.read(filename_queue)
    record_defaults = [[''] for _ in range(N_FEATURES)]
    # read in the 10 rows of data
    content = tf.decode_csv(value, record_defaults = record_defaults,field_delim = '\t') 


    # pack all 9 features into a tensor
    features = tf.stack(content[:N_FEATURES - 1])

    # assign the last column to label
    label = content[-1]

    # minimum number elements in the queue after a dequeue, used to ensure 
    # that the samples are sufficiently mixed
    # I think 10 times the BATCH_SIZE is sufficient
    min_after_dequeue = 10 * BATCH_SIZE

    # the maximum number of elements in the queue
    capacity = 20 * BATCH_SIZE

    # shuffle the data to generate BATCH_SIZE sample pairs
    data_batch, label_batch = tf.train.batch([features, label], batch_size=BATCH_SIZE, 
                                        capacity=capacity, min_after_dequeue = min_after_dequeue,
                                        allow_smaller_final_batch=True)

    return data_batch, label_batch
    # return features,label
def generate_batches(data_batch, label_batch):
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init_op)
        for _ in range(100): # generate 10 batches
            features, labels = sess.run([data_batch, label_batch])
            print features.shape
        coord.request_stop()
        coord.join(threads)
def main():
    data_batch, label_batch = batch_generator([DATA_PATH])
    generate_batches(data_batch, label_batch)

if __name__ == '__main__':
    main()
