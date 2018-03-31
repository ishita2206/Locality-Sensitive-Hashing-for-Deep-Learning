from NeuronLayer import *
import math
import random

class SoftMaxNeuronLayer(NeuronLayer):

	def __init__(self, prev_layer_size, layer_size, L2):
		super().__init__(prev_layer_size, layer_size, L2)

	def clone(self):
		copy = SoftMaxNeuronLayer(self.m_prev_layer_size, self.m_layer_size, self.L2_Lambda)
		copy.m_theta = self.m_theta
		copy.m_pos = self.m_pos
		return copy

	def weightInitialization(self):
		interval = 2 * math.sqrt(6.0 / (self.m_prev_layer_size + self.m_layer_size))
		return random.random() * (2*interval) - interval

	def activationFunction(self, input):
		sum = 0.0
		output = [None]*len(input)

		for idx in range(0, len(input)):
			output[idx] = math.exp(input[idx])
			sum = sum+output[idx]

		for idx in range(0, len(output)):
			output[idx] = output[idx]/sum

		return output

	def forwardPropagation(self, input):
		assert len(input) == len(self.m_prev_layer_size)
		self.m_input = input

		for jdx in range(0, self.m_layer_size):
			self.m_weightedSum[jdx] = 0

			for idx in range(0, self.m_prev_layer_size):
				self.m_weightedSum[idx] = self.m_weightedSum[idx]+ self.m_theta.getWeight(self.m_pos, jdx, idx) * self.m_input[idx]

			self.m_weightedSum[jdx] = self.m_weightedSum[jdx] + self.m_theta.getBias(self.m_pos, jdx)

		return activationFunction()

	def calculateDelta(self, prev_layer_delta):
		assert len(prev_layer_delta) == self.m_layer_size
		self.m_delta = prev_layer_delta
		return self.m_delta

	def calculateGradient(self):
		assert len(self.m_delta) == self.m_layer_size

		for idx in range(0, self.m_layer_size):
			for jdx in range(0, self.m_prev_layer_size):
				self.m_theta.stochasticGradientDescent(self.m_theta.weightOffset(self.m_pos, idx, jdx), self.m_delta[idx] * self.m_input[jdx] + self.L2_Lambda * self.m_theta.getWeight(self.m_pos, idx, jdx))

			self.m_theta.stochasticGradientDescent(self.m_theta.biasOffset(self.m_pos, idx), self.m_delta[idx])