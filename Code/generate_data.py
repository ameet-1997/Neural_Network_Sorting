import pandas as pd
import argparse
import numpy as np

"""Generates the data given the total number of 
elements to sort

If n elements, considers n-digit numbers
"""

def argparser():
    Argparser = argparse.ArgumentParser()
    Argparser.add_argument('--n', type=int, default=5, help='Number of elements')
    Argparser.add_argument('--range', type=int, default=100, help='Max Element')
    Argparser.add_argument('--train', type=int, default=1000, help='Train Data Size')
    Argparser.add_argument('--test', type=int, default=1000, help='Test Data Size')
    Argparser.add_argument('--train_name', type=str, default='../Data/train', help='Train File Name')
    Argparser.add_argument('--test_name', type=str, default='../Data/test', help='Test File Name')

    
    args = Argparser.parse_args()
    return args

def generate(args):
	train = np.random.randint(low=0, high=args.range, size=(args.train, args.n))
	train_sorted = np.sort(train)
	test = np.random.randint(low=0, high=args.range, size=(args.test, args.n))
	test_sorted = np.sort(test)

	# Column names
	cols = ['number'+str(i) for i in range(args.n)]

	# Write to file
	pd.DataFrame(train).to_csv(args.train_name+'_'+str(args.n)+'.csv', header=cols, index=False)
	pd.DataFrame(train_sorted).to_csv(args.train_name+'_'+str(args.n)+'_sorted'+'.csv', header=cols, index=False)
	pd.DataFrame(test).to_csv(args.test_name+'_'+str(args.n)+'.csv', header=cols, index=False)
	pd.DataFrame(test_sorted).to_csv(args.test_name+'_'+str(args.n)+'_sorted'+'.csv', header=cols, index=False)

def main(args):
	generate(args)



if __name__ == '__main__':
	args = argparser()
	main(args)