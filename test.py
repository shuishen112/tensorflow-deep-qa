import tensorflow as tf 


def test_variable_scope():
	with tf.variable_scope("foo"):
	    v = tf.get_variable("v", [1])
	with tf.variable_scope("foo", reuse=True):
	    v1 = tf.get_variable("v", [1])
	assert v1 is v
if __name__ == '__main__':
	test_variable_scope()
	# main()
