from HiddenLayer import *
import math
from Util import *
import random

class ReLUNeuronLayer(HiddenLayer):

	def __init__(self, prev_layer_size, layer_size, L2):
		super().__init__(prev_layer_size, layer_size, L2)

	def clone(self):
		copy = ReLUNeuronLayer(self.m_prev_layer_size, self.layer_size, self.L2_Lambda)
		copy.m_theta = self.m_theta
		copy.m_pos = self.m_pos

		return copy

	def weightInitialization(self):
		interval = 2 * math.sqrt(6.0 / (self.m_prev_layer_size + self.m_layer_size))
		return random.random() * (2*interval) - interval

	def activationFunction(self, input):
		output = [None]*len(input)

		for idx in range(0, len(output)):
			output[idx] = math.max(input[idx], 0.0)

		return output

	def derivative(self, input):
		if input > 0.0:
			return 1.0
		return 0.0
