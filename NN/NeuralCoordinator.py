from Util import *
from NeuralNetwork import *
import random
from threading import Thread

class NeuralCoordinator(object):
	def __init__(self, model_title, title, dataset, params, layers, hiddenLayers, L2, cf):
		self.m_modelTitle = model_title
		self.m_model_path = Util.DATAPATH + dataset + "/" + Util.MODEL + title + "_" + model_title;
		self.m_train_path = Util.DATAPATH + dataset + "/" + Util.TRAIN + title;
		self.m_test_path = Util.DATAPATH + dataset + "/" + Util.TEST + title;
		self.m_total_nodes = 0
		for layer in layers:
			self.m_total_nodes = self.m_total_nodes + layer.m_layer_size

		self.m_params = params
		self.m_networks = [None]*Util.LAYER_THREADS
		self.m_networks.append(NeuralNetwork(params, layers, hiddenLayers, L2, cf))

		for idx in range(0, Util.LAYER_THREADS):
			hiddenLayers1 = list()
			for e in hiddenLayers:
				hiddenLayers1.append(e.clone())

			layers1 = list()
			for e in hiddenLayers1:
				layers1.append(e)
			layers1.append(layers[-1].clone())
			self.m_networks.append(NeuralNetwork(params, layers1, hiddenLayers1, L2, cf))

	def initIndices(self, length):
		indices = list()
		for idx in range(0, length):
			indices.append(idx)
		return indices

	def shuffle(self, indices):
		for idx in range(0, len(indices)):
			rand = random.randrange(len(indices))
			value = indices[idx]
			indices[idx] = indices[rand]
			indices[rand] = value

	def test(self, data, labels):
		test_hashes = self.m_params.computeHashes(data)
		print("Finished Precomputing Hashes")
		print(self.m_networks[0].test(test_hashes, data, labels))

	@staticmethod
	def myRun(input_hashes, data, labels, start, end):
		for idx in range(start, end):
			self.network.execute(input_hashes[idx], data[idx], labels[idx], true)

	def train(self, max_epoch, data, labels, test_data, test_labels):
		assert len(data) == len(labels)
		assert len(test_data) == len(test_labels)

		input_hashes = self.m_params.computeHashes(data)
		print("Finished Precomputing Training Hashes")

		test_hashes = self.m_params.computeHashes(test_data)
		print("Finished Precomputing Testing Hashes")

		data_idx = self.initIndices(labels.shape[0])
		m_examples_per_thread = data.shape[0] / (Util.UPDATE_SIZE * Util.LAYER_THREADS)

		assert len(data_idx) == labels.shape[0]

		train_writer = open(self.m_train_path)
		test_writer = open(self.m_test_path)

		for epoch_count in range(0, max_epoch):
			self.m_params.clear_gradient()
			self.shuffle(data_idx) 
			count = 0
			while count < len(data_idx):
				threads = list()
				for network in self.m_networks:
					if count < len(data_idx):
						start = count
						count = min(len(data_idx), count+self.m_examples_per_thread)
						end = count

						t = Thread(target=NeuralCoordinator.myRun, args=(input_hashes, data, labels, start, end))
						t.start()
						threads.append(t)

				Util.join(threads)

				if epoch_count <= update_threshold and epoch_count % (epoch_count / 10 + 1 ) == 0:
					self.m_params.rebuildTables()

			epoch = self.m_params.epoch_offset()+epoch_count

			activeNodes = self.calculateActiveNodes(self.m_total_nodes * len(data.shape[0]))
			test_accuracy = self.m_networks[0].test(test_hashes, test_data, test_labels)
			print("Epoch :" + str(epoch) + "Accuracy : " + str(test_accuracy))

			test_writer.write(self.m_modelTitle + " " + str(epoch) + " " + str(activeNodes) + " " + str(test_accuracy))
			train_writer.write(self.m_modelTitle + " " + str(epoch) + " " + str(activeNodes) + " " + str(self.calculateTrainAccuracy(data.shape[0])))

			test_writer.flush()
			train_writer.flush()

			self.m_params.timeStep()

		test_writer.close()
		train_writer.close()
		self.save_model(max_epoch, self.m_model_path)



	def save_model(self, epoch, path):
		self.m_params.save_model(Util.writerBZ2(path), epoch)

	def calculateTrainAccuracy(self, size):
		count = 0
		for network in self.m_networks:
			count = count + network.m_train_correct
			network.m_train_correct = 0

		return count / size

	def calculateActiveNodes(self, total):
		active = 0

		for network in self.m_networks:
			active = active + network.calculateActiveNodes()

		return active / total

	def calculateMultiplications(self):

		total = 0
		for network in self.m_networks:
			total = total + network.calculateMultiplications()

		return total