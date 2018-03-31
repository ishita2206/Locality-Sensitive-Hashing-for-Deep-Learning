import numpy as np

class Histogram(object):
	def __init__(self):
		histogram = dict()
	def add(self, data):
		for value in data:
			if value not in self.histogram.keys():
				self.histogram[value] = 1
			else:
				self.histogram[value] = self.histogram[value]+1

	def thresholdSet(self, count):
		list_map = list(self.histogram.items())
		list_map = sorted(list_map, key=lambda x: x[1], reverse=True)
		count = min(count, len(list_map))

		retrived = set()

		for index in range(0, count):
			retrived.add((list_map[index])[0])

		return retrived

# def main():
# 	a = dict()
# 	a['a'] = 3
# 	a['c'] = 1
# 	a['b'] = 2

# 	print(type(list(a.items())[1]))
# 	t = list(a.items())
# 	t = sorted(t, key=lambda x: x[1], reverse=True)

# if __name__ == '__main__':
# 	main()