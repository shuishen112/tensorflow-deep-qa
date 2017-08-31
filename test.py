import tensorflow as tf 
import numpy as np 
b = np.random.rand(4,3)
a = np.random.rand(4,3) 

c = tf.cross(a,b)

with tf.Session() as sess:
	print sess.run(c)
