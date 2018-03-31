from HiddenLayer import *
from math import exp, pow

class LogisticNeuronLayer(HiddenLayer):
	def __init__(self, prev_layer_size, layer_size, L2):
		super().__init__(prev_layer_size, layer_size, L2)

	def clone():
		copy = LogisticNeuronLayer(m_prev_layer_size, m_layer_size, L2_Lambda)
		copy.m_theta = self.m_theta
		copy.m_pos = self.m_pos
		return copy

	##Random weight initialization

	def activationFunction(self, input):
		output = [None] * len(input)
		for idx in range(0, len(output)):
			output[idx] = 1.0 / (1.0 + exp(-input[idx]))
		return output

	def derivative(input):
		negative_exp = exp(-input)
		return negative_exp / pow((1+negative_exp), 2)

