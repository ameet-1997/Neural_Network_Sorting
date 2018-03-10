import tensorflow as tf

def model1(x, layers):
	for l in range(len(layers)):
		x = tf.layers.dense(x, units=layers[l], activation=tf.nn.relu)
	return x


def modified_sigmoid(x):
	return tf.sigmoid(x)-0.5