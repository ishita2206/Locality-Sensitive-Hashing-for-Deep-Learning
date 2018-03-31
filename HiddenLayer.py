import numpy as np
from NeuronLayer import NeuronLayer 
from abc import ABC

class HiddenLayer(NeuronLayer, ABC):
	def __init__(self, prev_layer_size, layer_size, L2):
		super().__init__(prev_layer_size, layer_size, L2)
		self.m_total_node_set = set()
		self.m_delta = [None]*self.m_layer_size

	def clone(self):
		pass

	def derivative(self, input):
		pass

	@classmethod
	def forwardPropagation(self, input, nn_node_set = None, hashes = None, training = False ):
		if nn_node_set == None and hashes == None and training == False and (type(input) == np.array or type(input) == list()):
			return self.forwardPropagation(Util.vectorize(input), training = False)
	
		elif nn_node_set == None and hashes == None and (type(input) == np.array or type(input) == list()):
			return self.forwardPropagation(Util.vectorize(input), training = training)
	
		elif nn_node_set == None and hashes == None and type(input) == np.ndarray:
			return self.forwardPropagation(input, nn_node_set = m_theta.retrieveNodes(m_pos, input), training = training)

		elif nn_node_set == None and type(hashes) == list and type(input) == np.ndarray:
			return self.forwardPropagation(input, nn_node_set = m_theta.retrieveNodes(m_pos, hashes), training = training)
		else:

			assert len(nn_node_set) <= self.m_layer_size
			assert len(input) == self.m_prev_layer_size

			self.m_input = input.flatten()
			self.m_node_set = nn_node_set;

			if training:
				self.m_total_nn_set_size = self.m_total_nn_set_size + len(self.m_node_set)
				m_total_multiplication = self.m_total_multiplication + len(self.m_node_set)*self.m_prev_layer_size

			self.m_weightedSum = np.zeros(self.m_weightedSum.shape) ##Check

			for idx in nn_node_set:
				self.m_weightedSum[idx] = self.m_theta.getWeightVector(self.m_pos, idx).dot(input) + self.m_theta.getBias(self.m_pos, idx)

			return activationFunction()

	@classmethod
	def calculateDelta(self, prev_layer_delta):

		self.m_delta = np.zeros(self.m_delta.shape)
		for idx in self.m_node_set:
			for jdx in range(0, len(prev_layer_delta), 1):
				self.m_delta[idx] = self.m_delta[idx] + self.m_theta.getWeight(self.m_pos+1, jdx, idx) * prev_layer_delta[jdx]
			self.m_delta[idx] = self.m_delta[idx] * self.derivative(self.m_weightedSum[idx])

		return m_delta

	@classmethod
	def calculateGradient(self):

		assert len(self.m_delta) == self.m_layer_size

		for idx in self.m_node_set:
			##Set node gradient
			for jdx in range(0, self.m_prev_layer_size, 1):
				self.m_theta.stochasticGradientDescent(self.m_theta.weightOffset(self.m_pos, idx, jdx), self.m_delta[idx]*self.m_input[jdx])
			self.m_theta.stochasticGradientDescent(self.m_theta.biasOffset(self.m_pos, idx), self.m_delta[idx])

	@classmethod
	def updateHashTables(size):

		print(str(self.m_pos) + ":" + str(self.m_total_nn_set_size / size))

