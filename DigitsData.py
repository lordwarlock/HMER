import pickle
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
from scipy import misc
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

class BalanceDigitsData(object):
	def __init__(self,data_path, valid_percentage = .1):
		with open(data_path,'rb') as fin:
			data = pickle.load(fin)
		self.data_size = len(data)
		self.valid_percentage = valid_percentage
		self.valid_dict = self.make_dict(data[:int(len(data) * valid_percentage)])
		self.train_dict = self.make_dict(data[int(len(data) * valid_percentage):])
	def make_dict(self,data):
		data_dict = dict()
		for img_label in data:
			label = np.argmax(img_label[1])
			image = img_label[0]
			if label in data_dict:
				data_dict[label].append(image)
			else:
				data_dict[label] = [image]
		return data_dict

	def get_valid(self):
		valid_size = int(self.data_size * self.valid_percentage)
		images = np.zeros((	2*valid_size,
							self.valid_dict[0][0].shape[0],
							self.valid_dict[0][1].shape[1]))
		labels = np.zeros((2*valid_size,len(self.valid_dict)))
		for idx in range(2*valid_size):
			random_label = np.random.randint(0,len(self.valid_dict))
			random_image = random.choice(self.valid_dict[random_label])
			images[idx,:,:] = random_image
			labels[idx,random_label] = 1.
		return images,labels
	def shuffle(self):
		pass
	def next_batch(self,batch_size):
		images = np.zeros((	batch_size,
							self.train_dict[0][0].shape[0],
							self.train_dict[0][1].shape[1]))
		labels = np.zeros((batch_size,len(self.train_dict)))
		for idx in range(batch_size):
			random_label = np.random.randint(0,len(self.train_dict))
			random_image = random.choice(self.train_dict[random_label])
			images[idx,:,:] = random_image
			labels[idx,random_label] = 1.
		return images,labels

class MnistDigitsData(object):
	def __init__(self):
		self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	def get_valid(self,size = 1000):
		data = self.mnist.train.next_batch(size)
		images = np.zeros((size,32,32))
		labels = data[1]
		for i in range(1000):
			images[i,:,:] = misc.imresize(np.reshape(data[0][i],(28,28)),(32,32))
		return images,labels
	def shuffle(self):
		pass
	def next_batch(self,batch_size):
		data = self.mnist.train.next_batch(batch_size)
		images = np.zeros((batch_size,32,32))
		labels = data[1]
		for i in range(batch_size):
			images[i,:,:] = misc.imresize(np.reshape(data[0][i],(28,28)),(32,32))
		return images,labels
if __name__ == '__main__':
	d=DigitsData('digits.data')
	d.next_batch(128)