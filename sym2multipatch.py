import pickle
import numpy as np
import random
class MultiClassData(object):
	def __init__(self,file_path):
		self.POS_PERCENTAGE = .4
		self.read_data(file_path)
	def read_data(self,file_path):
		with open(file_path,'rb') as fin:
			whole_data = pickle.load(fin)

		class_size = len(whole_data[0][1])
		self.CLASS_SIZE = class_size
		self.data_lists = [list() for i in range(class_size)]
		self.DATA_SHAPE = whole_data[0][0].shape
		for data in whole_data:
			class_id = np.ndarray.item(np.where(data[1])[0])
			self.data_lists[class_id].append(data[0])
	def next_batch(self,class_id,batch_size = 32):
		batch = np.zeros(tuple([batch_size]+list(self.DATA_SHAPE)))
		label = np.zeros((batch_size,2))
		for i in range(batch_size):
			if np.random.random() < self.POS_PERCENTAGE:
				#assign a positive data
				batch[i,:] = random.choice(self.data_lists[class_id])
				label[i,0] = 0
				label[i,1] = 1
			else:
				#assign a negative data
				while True:
					neg_class_id = np.random.randint(0,self.CLASS_SIZE)
					if neg_class_id != class_id:
						break	
				batch[i,:] = random.choice(self.data_lists[neg_class_id])
				label[i,0] = 1
				label[i,1] = 0
		return batch,label
if __name__ == '__main__':
	from scipy import misc
	mc  = MultiClassData('symbols1.data')
	batch,label = mc.next_batch(2)
	for i in range(label.shape[0]):
		misc.imsave(str(i)+'_'+str(label[i,:])+'.png',batch[i,:])