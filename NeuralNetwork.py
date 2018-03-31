class NeuralNetwork(object):
	def __init__(params, layers, hiddenLayers, L2, cf):
		self.L2_Lambda = L2
		self.m_cf = cf
		self.m_params = params
		self.m_hidden_layers = hiddenLayers
		self.m_layers = layers

		m_cost = None
		m_train_correct = None

	def calculateActiveNodes(self):
		total = 0
		for l in self.m_hidden_layers:
			total = total + l.m_total_nn_set_size
			l.m_total_nn_set_size = 0

		total = total + self.m_layers[len(self.m_layers) - 1].m_layer_size

		return total

	def calculateMultiplications(self):
		total = 0
		for l in self.m_hidden_layers:
			total = total + l.m_total_multiplication
			l.m_total_multiplication = 0

		total = total + self.m_layers[len(self.m_layers) - 1].numWeights()
		return total

	def test(self, input_hashes, data, labels):
		y_hat = [None]*len(labels)

		for idx in range(0, len(labels)):
			y_hat[idx] = self.forwardPropagation(data[idx], input_hashes[idx], false)

		return m_cf.accuracy(y_hat, labels)

	def getGradient(self,idx):
		return self.m_params.getGradient(idx)

	def copyTheta(self):
		return self.m_params.copy()

	def getCost(self):
		return self.m_cost

	def numTheta(self):
		return len(self.m_params)

	def getTheta(self, idx):
		return self.m_params.getTheta(idx)

	def setTheta(self, idx, value = None):
		if type(idx) == list or type(idx) == np.array:
			self.m_params.copy(idx)
		else:
			self.m_params.setTheta(idx, value)

	def computeHashes(self, data):
		return self.m_params.computeHashes(data)

	def updateHashTables(self, miniBatchSize):
		for e in self.m_hidden_layers:
			e.updateHashTables(miniBatchSize)

	def execute(self, hashes, input, labels, training):
		y_hat = self.forwardPropagation(input, hashes, training)
		self.backPropagation(y_hat, labels)

		self.m_train_correct = self.m_train_correct + self.m_cf.correct(y_hat, labels)

	def backPropagation(self, y_hat, labels):
		
		self.m_cost = self.m_cf.costFunction(y_hat, labels) + L2_Lambda*m_params.L2_regularization()
		
		outputLayer = self.m_layers[len(m_layers)-1]

		delta = self.m_cf.outputDelta(y_hat, labels, outputLayer)

		it = self.m_layers[-1]

		for i in range(len(m_layers)-1,-1,-1):
			self.m_layers[it].calculateGradient()

	def forwardPropagation(input, hashes, training):
		iterator = self.m_hidden_layers[0]

		data = iterator.forwardPropagation(input, hashes, training)
		for i in range(1, len(self.m_hidden_layers)):
			data = self.m_hidden_layers[i].forwardPropagation(data, training)

		return self.m_layers[-1].forwardPropagation(data)


