import numpy as np

class Pooling:
	def __init__(self):
		print ("Created object")

	@staticmethod
	def compress(size, data):
		compressSize = int(len(data) // size)
		# if len(data) % size > 0:
		# 	compressSize = compressSize + 1
		compressData = np.zeros(int(compressSize))
		# print(compressSize)
		for index in range(0,compressSize):
			offset = index*size
			# print(offset, offset+size)
			compressData[index] = Pooling.sum(data, offset, offset+size)
		return compressData

	@staticmethod
	def sum(data, startIndex, stopIndex):
		value = 0
		# print(startIndex, stopIndex, stopIndex-startIndex)
		startIndex = int(startIndex)
		stopIndex = int(stopIndex)
		if stopIndex > len(data):
			stopIndex = len(data)
		for index in range(startIndex, stopIndex):
			value = value + data[index]
		return value / (stopIndex-startIndex)

# def main():
# 	data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 	# p = Pooling()
# 	print(Pooling.compress(3,data))

# if __name__ == '__main__':
# 	main()
