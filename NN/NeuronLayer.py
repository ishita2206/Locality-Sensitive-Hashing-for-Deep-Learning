from abc import ABC

class NeuronLayer(ABC):
	def __init__(self, prev_layer_size, layer_size, L2):
		print("NeuronLayer Constructor")
		print(prev_layer_size, layer_size)
		self.m_pos = -1
		self.m_theta = None
		self.m_input = list()
		self.m_weightedSum = [None] * layer_size
		self.m_delta = list()

		self.m_prev_layer_size = prev_layer_size
		self.m_layer_size = layer_size
		self.L2_Lambda = L2

	def clone(self):
		pass

	def weightInitialization(self):
		pass

	def activationFunction(self, input = None):
		if input == None:
			return self.activationFunction(m_weightedSum)
		else :
			pass

	def numWeights(self):
		print(type(self.m_prev_layer_size))
		print(type(self.m_layer_size))
		return self.m_prev_layer_size * self.m_layer_size

	def numBias(self):
		return self.m_layer_size

	def forwardPropagation(self, input):
		pass

	def calculateDelta(self, prev_layer_delta):
		pass

	def calculateGradient(self):
		pass
