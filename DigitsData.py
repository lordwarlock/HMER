import pickle
import numpy as np
class DigitsData(object):
	def __init__(self,data_path, valid_percentage = .1):
		with open(data_path,'rb') as fin:
			data = pickle.load(fin)
			np.random.shuffle(data)
			self.valid = data[:int(len(data) * valid_percentage)]
			self.train = data[int(len(data) * valid_percentage):]
		self.shuffle()
	def shuffle(self):
		np.random.shuffle(self.train)
		self.next_batch_idx = 0
	def get_valid(self):
		length = len(self.valid)
		images = np.zeros((length,self.train[0][0].shape[0],self.train[0][0].shape[1]))
		labels = np.zeros((length,len(self.train[0][1])))
		for idx in range(length):
			images[idx,:,:] = self.valid[idx][0]
			labels[idx,:] = self.valid[idx][1]
		return images,labels
	def next_batch(self,batch_size):
		images = np.zeros((batch_size,self.train[0][0].shape[0],self.train[0][0].shape[1]))
		labels = np.zeros((batch_size,len(self.train[0][1])))
		if (self.next_batch_idx + batch_size) > len(self.train):
			leftover = self.next_batch_idx + batch_size - len(self.train)
			next_time = leftover
		else:
			leftover = 0
			next_time = self.next_batch_idx + batch_size
		batch_idx = 0
		for idx in range(self.next_batch_idx,min(self.next_batch_idx + batch_size, len(self.train))):
			images[batch_idx,:,:] = self.train[idx][0]
			labels[batch_idx,:] = self.train[idx][1]
			batch_idx += 1
		for idx in range(leftover):
			images[batch_idx,:,:] = self.train[idx][0]
			labels[batch_idx,:] = self.train[idx][1]
			batch_idx += 1
		self.next_batch_idx = next_time
		return images,labels


if __name__ == '__main__':
	d=DigitsData('digits.data')
	d.next_batch(128)