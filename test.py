import tensorflow as tf 

tf.flags.DEFINE_integer("embedding_dim",50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")

FLAGS = tf.flags.FLAGS

FLAGS._parse_flags()
print(("\nParameters:"))
for attr, value in sorted(FLAGS.__flags.items()):
		 print(("{}={}".format(attr.upper(), value)))
print((""))