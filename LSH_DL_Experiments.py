from MNISTDataSet import *
from CrossEntropy import *
from HiddenLayer import *
from NN_Parameters import *
from NeuralCoordinator import *
from NeuronLayer import *
from ReLuNeuronLayer import *
from SoftMaxNeuronLayer import *
from Util import *
import time

class LSH_DL_Experiments(object):
	def __init__(self):
		self.min_layers = 1
		self.max_layers = 1
		self.hidden_layer_size = 1000
		self.hidden_pool_size = self.hidden_layer_size * 0.1

		self.title = None
		self.dataset = None
		self.training_size = None
		self.test_size = None
		self.inputLayer = None
		self.outputLayer = None
		self.k = None
		self.b = 6
		self.L = 100

		self.max_epoch = 25
		self.L2_Lambda = 0.003
		self.hiddenLayers = None
		self.learning_rates = None
		self.size_limits = [0.05, 0.10, 0.25, 0.5, 0.75, 1.0]

		self.hidden_layers = None
		self.NN_layers = None

	def make_title(self):
		titleBuilder = ""
		titleBuilder = titleBuilder + self.dataset + "_LSH_"
		titleBuilder = titleBuilder + str(self.inputLayer) + "_"

		for idx in range(0, len(self.hiddenLayers)):
			titleBuilder = titleBuilder + str(self.hiddenLayers[idx]) + "_"

		titleBuilder = titleBuilder + str(self.outputLayer)
		return titleBuilder

	def testMNIST(self):
		self.dataset = "MNIST"
		self.training_size = 60000
		self.test_size = 10000
		self.inputLayer = 784
		self.outputLayer = 10
		self.k = 98
		self.learning_rates = [1e-2, 1e-2, 1e-2, 5e-3, 1e-3, 1e-3]

		training_label_path = "C:\\Users\\Ishita Doshi\\Desktop\\MNIST"
		training_image_path = "C:\\Users\\Ishita Doshi\\Desktop\\MNIST"
		test_label_path = "C:\\Users\\Ishita Doshi\\Desktop\\MNIST"
		test_image_path = "C:\\Users\\Ishita Doshi\\Desktop\\MNIST"

		training_images, training_labels = MNISTDataSet.loadDataSet(training_image_path, training_label_path, True)
		testing_images, testing_labels = MNISTDataSet.loadDataSet(test_image_path, test_label_path, False)

		self.execute(training_images, training_labels, testing_images, testing_labels)

	def construct(self, inputLayer, outputLayer):
		self.hidden_layers = list()
		self.hidden_layers.append(ReLUNeuronLayer(inputLayer, self.hiddenLayers[0], self.L2_Lambda))
		for idx in range(0, len(self.hiddenLayers)-1):
			self.hidden_layers.append(ReLUNeuronLayer(self.hiddenLayers[idx], self.hiddenLayers[idx+1], self.L2_Lambda))

		self.NN_layers = list()
		for layer in self.hidden_layers:
			self.NN_layers.append(layer)

		self.NN_layers.append(SoftMaxNeuronLayer(self.hiddenLayers[-1], outputLayer, self.L2_Lambda))

	def execute(self, training_data, training_labels, test_data, test_labels):
		assert len(self.size_limits) == len(self.learning_rates)

		for size in range(self.min_layers, self.max_layers+1):
			self.hiddenLayers = [None] * size
			sum_pool = [None]*size
			bits = [None]*size
			tables = [None]*size
			for idx in range(0, size):
				self.hiddenLayers[idx] = self.hidden_layer_size
				sum_pool[idx] = self.hidden_pool_size
				bits[idx] = self.b
				tables[idx] = self.L

			for idx in range(0, len(self.size_limits)):
				s1 = [None]*size
				for idx in range(0, size):
					s1[idx] = self.size_limits[idx]
					print(self.make_title())
					self.construct(self.inputLayer, self.outputLayer)

					try :
						parameters = NN_Parameters(self.NN_layers, sum_pool, bits, tables, self.learning_rates[idx], s1, Util.readerBZ2(Util.DATAPATH+dataset+"/"+Util.MODEL+title))
					except:
						parameters = NN_Parameters(self.NN_layers, sum_pool, bits, tables, self.learning_rates[idx], s1)

					NN = NeuralCoordinator(str(self.size_limits[idx]), self.title, self.dataset, parameters, self.NN_layers, self.hidden_layers, self.L2_Lambda, CrossEntropy())
					startTime = (time.time()*1000)
					NN.train(max_epoch, training_data, training_labels, test_data, test_labels)
					estimatedTime = (time.time()*1000 - startTime) / 1000
					print(estimatedTime)


def main():
	tester = LSH_DL_Experiments()
	tester.testMNIST()

if __name__ == '__main__':
	main()
