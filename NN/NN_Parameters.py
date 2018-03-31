from CosineDistance import *
from HashBuckets import *
import math
import numpy as np

class NN_Parameters():

	def __init__(self, NN_structure, poolDim, b, L, learning_rate, size_limit, reader = None):
		self.m_momentum_lambda = 0.50
		self.momentum_max = 0.90;
		self.momentum_rate = 1.00;

		self.m_epoch_offset = None
		self.m_weight_idx = None
		self.m_bias_idx = None
		self.m_size = 0
		self.m_layers_count = 0
		self.m_layer_row = None
		self.m_layer_col = None

		self.m_theta = None
		self.m_gradient = None

		self.m_momentum = None
		self.m_learning_rates = None

		self.m_layers = NN_structure
		self.m_learning_rate = learning_rate
		self.m_poolDim = poolDim
		self.m_b = b
		self.m_L = L
		self.m_size_limit = size_limit
		self.m_tables = list()
		self.construct(NN_structure)

		if reader is None :
			self.weight_initialization(NN_structure)
		else:
			self.m_epoch_offset = int(reader.readLine())
			self.load_model(NN_structure, reader, self.m_theta)
			self.load_model(NN_structure, reader, self.m_momentum)
			self.load_model(NN_structure, reader, self.m_learning_rates)
		
		self.createLSHTables(self.m_tables, poolDim, b, L, size_limit)
		print("Finished initializing paramters")

	def construct(self, NN_structure):
		t = len(NN_structure)
		self.m_weight_idx = [None] * t
		self.m_bias_idx = [None] * t
		self.m_layer_row = [None] * t
		self.m_layer_col = [None] * t

		for l in NN_structure:
			print(type(l))
			self.m_layer_row[self.m_layers_count] = l.m_layer_size
			self.m_layer_col[self.m_layers_count] = l.m_prev_layer_size

			self.m_weight_idx[self.m_layers_count] = self.m_size
			self.m_size = self.m_size + l.numWeights()
			self.m_bias_idx[self.m_layers_count] = self.m_size
			self.m_size = self.m_size + l.numBias()
			l.m_pos = self.m_layers_count
			self.m_layers_count = self.m_layers_count + 1
			l.m_theta = self

		self.m_theta = [None] * self.m_size
		self.m_gradient = [None] * self.m_size
		self.m_momentum = [None] * self.m_size
		self.m_learning_rates = [None] * self.m_size

	def copy(self, theta = None):
		if theta is None :
			temp = list()
			for idx in range(0, len(self.m_theta)):
				temp.append(self.m_theta[idx])
			return temp
		assert len(theta) == len(self.m_theta)
		for idx in range(0, len(theta)):
			m_theta[idx] = theta[idx]
		for idx in range(0, len(self.m_momentum)):
			self.m_momentum = 0
		for idx in range(0, len(self.m_learning_rates)):
			self.m_learning_rates = 0

	

	def epoch_offset(self):
		return self.m_epoch_offset

	def save_model(self, writer, epoch = None, array = None):
		if array is None:
			writer.write(str(epoch + self.m_epoch_offset))
			self.save_model(writer, None, self.m_theta)
			self.save_model(writer, None, self.m_momentum)
			self.save_model(writer, None, self.m_learning_rates)
			writer.close()
			return

		global_idx = -1

		for l in self.m_layers:

			for idx in range(0, l.m_layer_size):
				for jdx in range(0, l.m_prev_layer_size):
					global_idx = global_idx + 1
					writer.write(str(array[global_idx]) + " ")

			for idx in range(0, l.m_layers_size):
				global_idx = global_idx + 1
				writer.write(str(array[global_idx]) + " ")
		assert global_idx == m_size-1

	def load_model(self, NN_structure, reader, array):
		global_idx = -1
		for l in NN_structure:
			for idx in range(0, l.m_layer_size):
				node = reader.read().strip().split()
				assert len(node) == l.m_prev_layer_size
				for weight in node:
					global_idx = global_idx + 1
					array[global_idx] = float(weight)
			biases = reader.read().strip().split()
			assert len(biases) == l.m_layer_size
			for bias in biases:
				global_idx = global_idx + 1
				array[global_idx] = float(bias)

		assert global_idx == self.m_size-1

	def weight_initialization(self, NN_structure):
		global_idx = -1
		for l in NN_structure:
			for idx in range(0, l.m_layer_size):
				for jdx in range(0, l.m_prev_layer_size):
					global_idx = global_idx + 1
					self.m_theta[global_idx] = l.weightInitialization()

			global_idx = global_idx + l.m_layer_size
		assert global_idx == self.m_size-1

	def computeHashes(self, data):
		interval = len(data) / 10
		hashes = list()
		for idx in range(0, len(data)):
			if idx % interval == 0:
				print("Completed " + str(idx) + " / " + str(len(data)))
			hashes.append(self.m_tables[0].generateHashSignature(data[idx]))
		return hashes

	def getWeight(self, layer, node):
		assert layer >= 0 and layer < self.m_layers_count
		assert node >= 0 and node < m_layer_row[layer]

		return Util.vectorize(self.m_theta, self.m_weight_idx + node * self.m_layer_col[layer], self.m_layer_col[layer])

	def rebuildTables(self):
		global_idx = 0
		for layer_idx in range(0, self.m_layers_count-1):
			self.m_tables[layer_idx].clear()
			for idx in range(0, self.m_layer_row[layer_idx]):
				self.m_tables[layer_idx].LSHAdd(idx, Util.vectorize(self.m_theta, global_idx, m_layer_col[layer_idx]))
				global_idx = global_idx + self.m_layer_col[layer_idx]
			global_idx = global_idx + self.m_layer_row[layer_idx]

	def createLSHTables(self, tables, poolDim, b, L, size_limit):
		global_idx = 0
		for layer_idx in range(0, self.m_layers_count - 1):
			table = HashBuckets(size_limit[layer_idx] * self.m_layer_row[layer_idx], poolDim[layer_idx], L[layer_idx], CosineDistance(b[layer_idx], L[layer_idx], self.m_layer_col[layer_idx]/poolDim[layer_idx]))
			for idx in range(0, self.m_layer_row[layer_idx]):
				table.LSHAdd(idx, Util.vectorize(self.m_theta, global_idx, self.m_layer_col[layer_idx]))
				global_idx = global_idx + self.m_layer_col[layer_idx]
			tables.append(table)
			global_idx = global_idx + self.m_layer_row[layer_idx]

	def retrieveNodes(self, layer, input):
		return self.m_tables[layer].histogramLSH(input)
	
	def timeStep(self):
		self.m_momentum_lambda = self.m_momentum_lambda * self.momentum_rate
		self.m_momentum_lambda = min(self.m_momentum_lambda, self.momentum_max)

	def size(self):
		return self.m_size

	def getGradient(self, idx):
		assert idx >=0 and idx < len(self.m_theta)
		return self.m_momentum[idx] / self.m_learning_rate

	def getTheta(self, idx):
		assert idx >=0 and idx < len(self.m_theta)
		return self.m_theta[idx]

	def setTheta(self, idx, value):
		assert idx >=0 and idx < len(self.m_theta)
		self.m_theta[idx] = value

	def getWeight(self, layer, row, col):
		assert layer >= 0 and layer < self.m_layers_count
		assert row >= 0 and row < self.m_layer_row[layer]
		assert col >= 0 and col < self.m_layer_col[layer]

		idx = row * self.m_layer_col[layer] + col
		return self.m_theta[self.m_weight_idx[layer] + idx]

	def getWeightVector(self, layer, node_idx):
		assert layer >= 0 and layer < self.m_layers_count
		assert node_idx >= 0 and row < self.m_layer_row[layer]
		
		return Util.vectorize(self.m_theta, self.m_weight_idx[layer]+node_idx*self.m_layer_col[layer], self.m_layer_col[layer])

	def setWeight(self, layer, row, col, value):
		assert layer >= 0 and layer < self.m_layers_count
		assert row >= 0 and row < self.m_layer_row[layer]
		assert col >= 0 and col < self.m_layer_col[layer]

		idx = row * self.m_layer_col[layer] + col
		self.m_theta[self.m_weight_idx[layer]+idx] = value

	def getBias(self, layer, idx):
		assert layer >= 0 and layer < self.m_layers_count
		assert idx >= 0 and idx < self.m_layer_row[layer]

		return self.m_theta[self.m_bias_idx[layer]+idx]

	def setBias(self, layer, idx, value):
		assert layer >= 0 and layer < self.m_layers_count
		assert idx >= 0 and idx < self.m_layer_row[layer]

		self.m_theta[self.m_bias_idx[layer] + idx] = value

	def L2_regularization(self):
		L2 = 0.0
		for layer_idx in range(0, self.m_layers_count):
			for idx in range(self.m_weight_idx[layer_idx], self.m_bias_idx[layer_idx]):
				L2 = L2 + math.pow(self.m_theta[idx], 2)

		return 0.5 * L2

	def weightOffset(self, layer, row, column):
		assert layer >= 0 and layer < self.m_layers_count
		assert row >= 0 and row < self.m_layer_row[layer]
		assert col >= 0 and col < self.m_layer_col[layer]

		idx = row * self.m_layer_col[layer] + column
		return self.m_weight_idx[layer] + idx 

	def biasOffset(self, layer, idx):
		assert layer >= 0 and layer < self.m_layers_count
		assert idx >= 0 and idx < self.m_layer_row[layer]

		return self.m_bias_idx[layer] + idx

	def stochasticGradientDescent(self, idx, gradient):
		self.m_gradient[idx] = gradient
		self.m_learning_rates[idx] = self.m_learning_rates[idx] + math.pow(gradient, 2)
		learning_rate = self.m_learning_rate / (1e-6 + math.sqrt(self.m_learning_rates[idx]))
		self.m_momentum[idx] = self.m_momentum[idx] * self.m_momentum_lambda
		self.m_momentum[idx] = self.m_momentum[idx] + learning_rate * gradient
		self.m_theta[idx] = self.m_theta[idx] - self.m_momentum[idx]

	def clear_gradient(self):
		for idx in range(0, len(self.m_gradient)):
			self.m_gradient[idx] = 0

	def print_active_nodes(self, dataset, filename, threshold):
		linesize = 25

		writer = open(Util.DATAPATH + dataset + "/" + "_" + filename)
		string = "["
		count = 0
		for layer in range(0, len(self.m_layer_row)):
			layer_size = self.m_layer_row[layer]
			grad_count = [None]*layer_size
			for idx in range(0, layer_size):
				pos = idx * self.m_layer_col[layer]
				for jdx in range(0, m_layer_col[layer]):
					if math.abs(self.m_gradient[pos+jdx] > 0):
						grad_count[idx] = grad_count[idx] + 1

			for idx in range(0, layer_size):
				if grad_count[idx] >= threshold * layer_size:
					value = 1
				else:
					value = 0

				string = string + str(value)

				if not(layer == len(self.m_layer_row)-1 and idx == layer_size -1) :
					if count <= linesize:
						string = string + ", "
					else:
						string = string + "\n"
						count = 0
				count = count + 1
		string = string + "]\n"
		writer.write(string)
		writer.flush()
		writer.close()