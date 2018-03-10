import pandas as pd 
from arch import model1
import argparse
import numpy as np
import tensorflow as tf

class NeuralNetwork():

	def __init__(self, args):
		self.args = args
		np.random.seed(1234)
		self.display_step = 1

		# Placeholders representing input and expected output
		self.input = tf.placeholder(tf.float32, shape=(None, self.args.n), name='Unsorted_Numbers')
		self.output = tf.placeholder(tf.float32, shape=(None, self.args.n), name='Sorted_Numbers')
		self.batch_size = self.args.batch_size

		# Training and Test data
		self.train = np.array(pd.read_csv(self.args.train_name+'_'+str(self.args.n)+'.csv'))
		self.train_labels = np.array(pd.read_csv(args.train_name+'_'+str(args.n)+'_sorted'+'.csv'))
		self.test = np.array(pd.read_csv(self.args.test_name+'_'+str(self.args.n)+'.csv'))
		self.test_labels = np.array(pd.read_csv(self.args.test_name+'_'+str(self.args.n)+'_sorted'+'.csv'))

	# Training the model
	def train_model(self):

		# Get the output of the network
		output = model1(self.input, self.args.sizes)

		# Build the loss
		loss = self.loss(output,self.output)

		# Optimize
		optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr).minimize(loss)

		# Initialize variables
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		for epoch in range(self.args.epochs):
			# Load the batches
			total_batch = int(len(self.train) / self.batch_size)
			perm = np.random.permutation(self.train.shape[0])
			x_batches = np.array_split(self.train[perm], total_batch)
			y_batches = np.array_split(self.train_labels[perm], total_batch)

			# Average loss
			avg_loss = 0

			# Train
			for b in range(total_batch):
				feed_dict = {self.input:x_batches[b], self.output:y_batches[b]}
				_, c = sess.run([optimizer, loss], feed_dict=feed_dict)

				avg_loss += c / total_batch
			if epoch % self.display_step == 0:
				print("Epoch: %2d" % (epoch+1) + ", cost={:.9f}".format(avg_loss))

	def loss(self, x, y):
		return tf.reduce_mean(tf.squared_difference(x,y))


def comma_list(string):
    return string.split(',')

def comma_int_list(string):
    return list(map(int,string.split(',')))

def argparser():
    Argparser = argparse.ArgumentParser()
    Argparser.add_argument('--n', type=int, default=5, help='Number of elements')
    Argparser.add_argument('--sizes', type=comma_int_list, default=[5,5], help='Sizes of hidden layers')
    Argparser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    Argparser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    Argparser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    Argparser.add_argument('--train_name', type=str, default='../Data/train', help='Train File Name')
    Argparser.add_argument('--test_name', type=str, default='../Data/test', help='Test File Name')

    args = Argparser.parse_args()
    return args

def main(args):
	# Instantiate an object
	nn = NeuralNetwork(args)
	nn.train_model()

if __name__ == '__main__':
	args = argparser()
	main(args)