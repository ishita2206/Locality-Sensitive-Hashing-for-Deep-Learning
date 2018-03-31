import random
import time
import bz2
import numpy as np

class Util(object):
	DATAPATH = "../data/"
	MODEL = "Models/"
	TRAIN = "Train/"
	TEST = "Test/"

	INT_SIZE = 32
	LAYER_THREADS = 1
	UPDATE_SIZE = 10
	def __init__(self):

		random.seed(time.time())

	@staticmethod
	def randInt(self, min, max):
		return random.randrange(min, max)
	
	@staticmethod
	def randBoolean(probability):
		return random.random() < probability

	@staticmethod
	def writerBZ2(path):
		return bz2.open(path, "w")

	@staticmethod
	def readerBZ2(path):
		return bz2.open(path, "r")

	@staticmethod
	def byteReaderBZ2(path):
		return bz2.open(path, "r")

	@staticmethod
	def vectorize(data, offset = 0, length = 0):
		print(type(data), len(data))
		if offset == 0 and length == 0:
			return Util.vectorize(data, 0, len(data))
		vector = np.zeros((length,1))
		for idx in range(0, length):
			vector[idx,0] = data[offset+idx]

		return vector

	@staticmethod
	def mean_normalization(sum, data_list):
		meanVector = np.ndarray((len(sum),1))
		for idx in range(0, len(sum)):
			meanVector[idx,0] = sum[idx]

		meanVector = meanVector / len(data_list)

		for d in data_list:
			d = d - meanVector

		return data_list

	@staticmethod
	def range_normalization(min, max, data_list):
		min_vector = np.ndarray((len(min),1))
		for idx in range(0, len(min)):
			min_vector[idx] = min[idx]

		max_vector = np.ndarray((len(max),1))
		for idx in range(0, len(max)):
			max_vector[idx] = max[idx]

		range = max_vector - min_vector

		for d in data_list:
			d = d/range;

		return data_list

	@staticmethod
	def gradient_check(NN, data, labels, num_checks):
		input_hashes = NN.computeHashes(data)

		delta = 0.0001

		max = 0.0
		original_params = NN.copyTheta()
		
		for idx in range(0, num_checks):

			randData = Util.randInt(0, len(labels))
			randIdx = Util.randInt(0, NN.numTheta())

			theta = NN.getTheta(randIdx)

			NN.execute(input_hashes[randData], data[randData], labels[randData], false)
			gradient = NN.getGradient(randIdx)
			NN.setTheta(original_params)

			NN.setTheta(randIdx, theta-delta)
			NN.execute(input_hashes[randData], data[randData], labels[randData], false)
			J0 = NN.getCost()
			NN.setTheta(original_params)

			NN.setTheta(randIdx, theta+delta)
			NN.execute(input_hashes[randData], data[randData], labels[randData], false)
			J1 = NN.getCost()
			NN.setTheta(original_params)

			est_gradient = (J1-J0) / (2*delta)
			error = math.abs(gradient - est_gradient)
			print("Error: " + str(error) + " Gradient: " + str(gradient) + " Est.Gradient: " + str(est_gradient));
			max = math.max(max, error);

		return max

	@staticmethod
	def join(threads):
		for t in threads:
			t.join()

		return threads