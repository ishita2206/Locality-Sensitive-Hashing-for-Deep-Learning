import numpy as np
from mnist import MNIST

class MNISTDataSet(object):
	def __init__(self):
		LABEL_MAGIC = 2049
		IMAGE_MAGIC = 2051

	@staticmethod
	def loadDataSet(label_path, image_path, isTrain):
		mndata = MNIST(label_path)
		if isTrain :
			images,labels = mndata.load_training()
			temp = np.ndarray((60000, 784))
		else:
			images, labels = mndata.load_testing()
			temp = np.ndarray((10000, 784))
		labels = list(labels)
		for idx in range(0, len(images)):
			for jdx in range(0, 784):
				temp[idx][jdx] = images[idx][jdx] / 255.0

		return temp, labels

# def main():
# 	path = "C:\\Users\\Ishita Doshi\\Desktop\\MNIST"
# 	# print(path)
# 	MNISTDataSet.loadDataSet(path, path)


# if __name__ == '__main__':
# 	main()