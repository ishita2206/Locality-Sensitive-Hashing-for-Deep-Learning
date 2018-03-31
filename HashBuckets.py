import numpy as np
from LSH import *
from Histogram import *
from Pooling import *

class HashBuckets(object):
	def __init__(self, sizeLimit, poolDim, L, hashFunction):
		self.m_hashFunction = hashFunction
		self.m_poolDim = poolDim
		self.m_nn_sizeLimit = sizeLimit
		self.m_L = L
		self.m_Tables = list()
		self.m_bucket_hashes = list()
		self.construct()

	def construct(self):
		for i in range(0, self.m_L):
			self.m_Tables.append(dict())
			self.m_bucket_hashes.append(dict())

	def clear(self):
		self.m_Tables.clear()
		self.m_bucket_hashes.clear()

	def LSHAdd(self, recIndex, data):
		if type(data) is np.ndarray:
			return self.LSHAdd(recIndex, self.generateHashSignature(data))
		assert len(data) == self.m_L
		for index in range(0, self.m_L):
			if not(data[index] in self.m_Tables[index].keys()):
				s = set()
				s.add(recIndex)
				self.m_Tables[index][data[index]] = s
				self.m_bucket_hashes[index][data[index]] = data
			else:
				self.m_Tables[index][data[index]].add(recIndex)

	def LSHUnion(self, data):
		if type(data) is np.ndarray:
			return self.LSHUnion(self.generateHashSignature(data))
		assert len(data) == self.m_L
		retrieved = set()
		for index in range(0, self.m_L) and len(retrieved) < self.m_nn_sizeLimit :
			if self.m_Tables[index][data[index]] :
				retrieved.add(self.m_Tables[index][data[index]])

		return retrieved

	def histogramLSH(self, data):
		if type(data) is np.ndarray:
			return self.histogramLSH(self.generateHashSignature(data))
		assert len(data) == self.m_L
		hist = Histogram()
		for index in range(0, self.m_L):
			if self.m_Tables[index][data[index]] :
				hist.add(self.m_Tables[index][data[index]])

		return hist.thresholdSet(self.m_nn_sizeLimit)

	def generateHashSignature(self, data):
		return self.m_hashFunction.hashSignature(Pooling.compress(self.m_poolDim, data))