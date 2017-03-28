from __future__ import division
import pickle
import glob
import scipy.io as sio
import numpy as np
from scipy import misc
from skimage.transform import resize
class ImageGoldPair(object):
	def __init__(self,image,gold,window_radius = 10,neg_radius = 10):
		safe_radius = 3 * max(window_radius,neg_radius)
		self.safe_radius = safe_radius
		#print("safe_radius:",safe_radius)
		self.image = image
		self.gold = gold
		#self.image = resize(self.image,np.array(self.image.shape)//4)
		#self.gold = resize(self.gold,np.array(self.gold.shape)//4)
		#self.gold = self.gold > 0
		self.image = np.pad(self.image,((safe_radius,safe_radius),(safe_radius,safe_radius)),
				 'constant',constant_values=0)
		self.gold = np.pad(self.gold,((safe_radius,safe_radius),(safe_radius,safe_radius)),
				 'constant',constant_values=0)
		gold_xs,gold_ys = self.get_gold_sub(self.gold)
		#print('ImageGoldPair',self.image.dtype,self.gold.dtype)
		self.gold_coord = np.c_[gold_xs,gold_ys]
		#print(self.gold.shape)
	def get_gold_sub(self,gold):
		tmp_sub = np.where(gold == 255)
		return tmp_sub[0], tmp_sub[1]
class RealDataBatchGenerator(object):
	def __init__(self,window_radius,neg_radius = 10,
				shaking = 0,image_path = '../../data/neuron/big_dev',
				random_brightness = 50,
				random_pos_br_l = 0.5,
				random_pos_br_h = 1.0,
				random_neg_br_l = .5,
				random_neg_br_h = 1.0,
				random_contrast_l = 0.8,
				random_contrast_h = 1.2):
		self.data = []
		self.neg_radius = neg_radius
		self.shaking = shaking
		self.noise_mean = 40
		self.noise_std = 10
		self.random_brightness = random_brightness
		self.random_contrast_l = random_contrast_l
		self.random_contrast_h = random_contrast_h
		self.random_pos_br_h = random_pos_br_h
		self.random_pos_br_l = random_pos_br_l
		self.random_neg_br_h = random_neg_br_h
		self.random_neg_br_l = random_neg_br_l
		print("Loading Neuron Images")
		self.window_radius = window_radius
		self.load_symbol_images(image_path,window_radius)
		#self.load_images_and_swc(image_path,swc_root,window_radius)

	def load_symbol_images(self,image_path,window_radius):
		with open(image_path,'rb') as fi:
			images = pickle.load(fi)
			for idx in range(len(images)):
				image = np.uint8(images[idx][0])
				gold = np.uint8(images[idx][0])
				self.data.append(ImageGoldPair(image,gold,window_radius = window_radius,neg_radius = self.neg_radius))

	def next_batch(self,batch_size,
						pos_percent = .5):
		batch = np.zeros((batch_size,self.window_radius * 2 + 1,self.window_radius * 2 + 1,1))
		label = np.zeros((batch_size,2))
		for i in range(batch_size):
			curr_image_gold = np.random.choice(self.data)
			#print(curr_image_gold)
			if np.random.random() < pos_percent:
				patch,curr_label = self.pos_patch(curr_image_gold)
				error_info = "pos "+str(patch.shape)
			else:
				patch,curr_label = self.neg_patch(curr_image_gold)
				error_info = "neg "+str(patch.shape)
			try:
				batch[i,:,:,0] = patch
			except:
				print(error_info)
				raise
			if curr_label:
				label[i,1] = 1
			else:
				label[i,0] = 1
		#print(np.sum(label[:,1]))
		return batch/255.0,label

	def pos_patch(self,curr_image_gold):
		total,_ = curr_image_gold.gold_coord.shape
		selected = np.random.randint(0,total)
		x = curr_image_gold.gold_coord[selected,0]
		y = curr_image_gold.gold_coord[selected,1]
		#print(curr_image_gold.gold[x,y,z])
		non_shaking_label = curr_image_gold.gold[x,y]
		if np.random.random() < 0.5:
			x = x + np.random.randint(-self.shaking,self.shaking+1)
		else:
			y = y + np.random.randint(-self.shaking,self.shaking+1)

		if x - self.window_radius < 0 \
			or y - self.window_radius < 0:
			print(curr_image_gold.gold_coord[selected,0],curr_image_gold.gold_coord[selected,1])
		curr_patch = curr_image_gold.image[x - self.window_radius : x + self.window_radius + 1,
										   y - self.window_radius : y + self.window_radius + 1]

		noise = np.random.normal(self.noise_mean,self.noise_std,(1+2*self.window_radius,1+2*self.window_radius))
		#brightness adjust
		curr_patch = np.float32(curr_patch)
		curr_patch += noise
		curr_patch += np.random.random() * self.random_brightness
		curr_patch *= np.random.random() * (self.random_pos_br_h - self.random_pos_br_l) + self.random_pos_br_l
		curr_patch[curr_patch > 255] = 255
		curr_patch[curr_patch < 0] = 0
		#contrast adjust
		curr_patch = (curr_patch - np.mean(curr_patch)) \
			* (np.random.random() * (self.random_contrast_h - self.random_contrast_l) + self.random_contrast_l)\
			+ np.mean(curr_patch)
		curr_patch[curr_patch > 255] = 255
		curr_patch[curr_patch < 0] = 0
		#curr_patch = (curr_patch - np.mean(curr_patch))/(np.std(curr_patch) + 1e-6) * 255
		return curr_patch, non_shaking_label#curr_image_gold.gold[x,y,z]

	def neg_patch(self,curr_image_gold):
		total,_ = curr_image_gold.gold_coord.shape
		#print(total,curr_image_gold.gold_coord.shape)
		selected = np.random.randint(0,total)
		if np.random.random() > .5:
			x = curr_image_gold.gold_coord[selected,0] + np.random.randint(- self.neg_radius, self.neg_radius + 1)
			y = curr_image_gold.gold_coord[selected,1] + np.random.randint(- self.neg_radius, self.neg_radius + 1)

			if x - self.window_radius < 0 \
				or y - self.window_radius < 0:
				print(curr_image_gold.gold.shape,self.window_radius,self.neg_radius)
				print(curr_image_gold.gold_coord[selected,0],curr_image_gold.gold_coord[selected,1])
		else:
			size_x,size_y = curr_image_gold.image.shape
			safe_radius = curr_image_gold.safe_radius
			x = np.random.randint(safe_radius,size_x - safe_radius)
			y = np.random.randint(safe_radius,size_y - safe_radius)
		curr_patch = curr_image_gold.image[x - self.window_radius : x + self.window_radius + 1,
										   y - self.window_radius : y + self.window_radius + 1]

		noise = np.random.normal(self.noise_mean,self.noise_std,(1+2*self.window_radius,1+2*self.window_radius))
		#brightness adjust
		curr_patch = np.float32(curr_patch)
		curr_patch += noise
		curr_patch += np.random.random() * self.random_brightness
		curr_patch *= np.random.random() * (self.random_neg_br_h - self.random_neg_br_l) + self.random_neg_br_l
		curr_patch[curr_patch > 255] = 255
		curr_patch[curr_patch < 0] = 0
		#contrast adjust
		curr_patch = (curr_patch - np.mean(curr_patch)) \
			* (np.random.random() * (self.random_contrast_h - self.random_contrast_l) + self.random_contrast_l)\
			+ np.mean(curr_patch)
		curr_patch[curr_patch > 255] = 255
		curr_patch[curr_patch < 0] = 0
		#curr_patch = (curr_patch - np.mean(curr_patch))/(np.std(curr_patch) + 1e-6) * 255
		return curr_patch, curr_image_gold.gold[x,y]



if __name__ == '__main__':
	ig = RealDataBatchGenerator(15,image_path = 'symbols1.data')
	for i in range(16):
		patch,label = ig.next_batch(1)
		print(np.max(patch))
		misc.imsave('p'+str(i)+'_0'+str(label[0])+'.png',patch[0,:,:,0] * 255)
