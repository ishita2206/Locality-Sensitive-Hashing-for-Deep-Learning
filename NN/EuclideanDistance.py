import numpy as np
from math import sqrt

class EuclideanDistance(object):

	@staticmethod
	def distance(x, y):
		if type(x) is np.ndarray and type(y) is np.ndarray:
			return np.linalg.norm(x-y)
		assert len(x) == len(y)
		dist = 0
		x_mag = 0
		y_mag = 0

		for idx in range(0, len(x)):
			x_mag = x_mag + pow(x[idx],2)
			y_mag = y_mag + pow(y[idx],2)

		x_mag = sqrt(x_mag)
		y_mag = sqrt(y_mag)

		for idx in range(0, len(x)):
			dist = dist + pow((x[idx]/x_mag) - (y[idx]/y_mag), 2)

		return sqrt(dist)

# def main():
# 	A = np.ndarray((3,2))
# 	A[0] = [1, 2]
# 	A[1] = [2, 3]
# 	A[2] = [4, 7]

# 	B = np.ndarray((3,2))
# 	B[0] = [3, 5]
# 	B[1] = [7, 10]
# 	B[2] = [3, 1]

# 	t = EuclideanDistance()

# 	print(t.distance(A[2],B[1]))

# if __name__ == '__main__':
# 	main()