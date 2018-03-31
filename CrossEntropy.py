from math import log
# import NeuronLayer
from ICostFunction import *

class CrossEntropy(ICostFunction):
	def __init__(self):
		pass

	def max_idx(self, array):
		max_index = 0
		max_value = None
		for index in range(0, len(array)):
			if max_value is None or max_value < array[index]:
				max_index = index
				max_value = array[index]

		return max_index

	def correct(self, y_hat, labels):
		if max_idx(y_hat) == int(labels):
			return 1.0
		return 0.0

	def accuracy(self, y_hat, labels):
		correct = 0
		for index in range(0, len(labels)):
			if(max_idx(y_hat[idx]) == int(labels[index])):
				correct = correct + 1

		return correct / len(labels)

	def costFunction(self, y_hat, labels):
		return log(y_hat[int(labels)])

	def outputDelta(self, y_hat, labels, l):
		delta = deepcopy(y_hat)
		delta[int(labels)] = delta[int(labels)] - 1
		return delta