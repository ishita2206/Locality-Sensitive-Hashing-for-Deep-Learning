import numpy as np

class RandomProjection(object):

	def __init__(self, hashes, projection_matrix, query):
		# print(type(projection_matrix[0]), type(hashes), type(query))
		# print(len(hashes))
		self.m_projection_matrix = projection_matrix
		self.m_query = query
		self.m_hashes = hashes

	def run(self):
		hash_index = -1
		for projection in self.m_projection_matrix:
			# print(projection.shape[1], self.m_query.shape[0])
		
			assert projection.shape[1] == self.m_query.shape[0]
			# if type(projection) == list:
			# 	assert len(self.m_query) == 1
			# else:
			# 	assert len(projection) == 1 #projection.columns = m_query.rows
			dotProduct = projection.dot(self.m_query)
			# print(dotProduct)
			signature = 0
			for index in range(0, len(dotProduct)):
				signature |= self.sign(dotProduct[index])
				# print(dotProduct[index], self.sign(dotProduct[index]), signature)
				signature = signature << 1
			hash_index = hash_index+1
			# print(hash_index)
			# print(len(self.m_hashes))
			self.m_hashes[hash_index] = signature
		return self.m_hashes

	def sign(self, value):
		if value > 0:
			return 1
		return 0

# def main():
# 	projection_matrix = np.array([[1, 3, 5, 7],[2,4,6,8]])
# 	query = np.array([[3, 4, 5, 7],[1,2,3,4]])
# 	hashes = np.array([0,0])

# 	q = RandomProjection(hashes,projection_matrix, query.T)
# 	print(q.run())

# if __name__ == '__main__':
# 	main()