import numpy as np 
from random import randrange
from RandomProjection import *
from math import sqrt, cos, pi, floor
from Util import *
from LSH import *

class CosineDistance(LSH):
	def __init__(self, b, L, d):
		self.m_L = L
		self.randomMatrix = list()
		self.hashes = [None]*L
		for index in range(0, self.m_L):
			self.randomMatrix.append(np.random.randn(b,floor(d)))
	
	def hashSignature(self, data):
		return RandomProjection(self.hashes, self.randomMatrix, data).run()

	@staticmethod
	def distance(x,y):
		if type(x) is np.ndarray:
			return 1- (x.dot(y)/(np.linalg.norm(x) * np.linalg.norm(y)))
		assert len(x) == len(y)

		dp = 0
		x_norm = 0
		y_norm = 0

		for index in range(0, len(x)):
			dp = dp + x[index]*y[index]
			x_norm = x_norm + pow(x[index],2)
			y_norm = y_norm + pow(y[index],2)

		x_norm = sqrt(x_norm)
		y_norm = sqrt(y_norm)

		return 1 - (dp / (x_norm*y_norm))

	@staticmethod
	def dotProductDistance(x,y,b):
		numInts = b// Util.INT_SIZE;
		assert len(x) == numInts
		assert len(y) == numInts

		dp = 0
		x_norm = 0
		y_norm = 0

		for index in range(0, len(x)):
			dp = dp + CosineDistance.count(x[index]&y[index])
			x_norm = x_norm + CosineDistance.count(x[index])
			y_norm = y_norm + CosineDistance.count(y[index])

		x_norm = sqrt(x_norm)
		y_norm = sqrt(y_norm)

		return 1 - (dp / (x_norm*y_norm))

	@staticmethod
	def hammingDistance(x,y,b):
		numInts = b // Util.INT_SIZE
		numBits = b % Util.INT_SIZE

		if numBits == 0 :
			numBits = Util.INT_SIZE
		
		bitMask = int(pow(2, numBits) - 1)

		assert len(x) == numInts
		assert len(y) == numInts

		hammingDistance = 0

		for index in range(0, len(x)-1):
			hammingDistance = hammingDistance + CosineDistance.count(x[index] ^ y[index])

		temp = CosineDistance.count(x[-1]&bitMask) ^ (y[-1]&bitMask)
		hammingDistance = hammingDistance + temp

		return 1- cos(hammingDistance * pi / b)

	@staticmethod
	def count(value):
		cou = 0
		for index in range(0, Util.INT_SIZE):
			count = count + (value&1)
			value = value >> 1

		return count
