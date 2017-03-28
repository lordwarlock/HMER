import tensorflow as tf
import pickle
import numpy as np
import random
from skimage.morphology import binary_dilation
from skimage.util import random_noise
from scipy import misc
class SymbolRecognition(object):
	def __init__(self,sess,model_path = None,symbols_path = 'symbols.txt',trainflag = True):
		self.sess = sess
		self.random_brightness = .2
		self.random_pos_br_l = 0.1
		self.random_pos_br_h = 1.0
		self.random_neg_br_l = 1.0
		self.random_neg_br_h = 1.5
		self.random_contrast_l = 0.2
		self.random_contrast_h = 1.8
		self.trainflag = trainflag
		self.inference()
		self.saver = tf.train.Saver()
		symbol_list = []
		with open(symbols_path,'r') as fin:
			for line in fin:
				symbol_list.append(line.strip())
		self.symbol_list = symbol_list
		if model_path is not None:
			self.saver.restore(sess,model_path)
	
	def batch_norm_layer(self,inputs, decay = 0.9):
		is_training = self.trainflag
		epsilon = 1e-3
		scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
		beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
		pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
		pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
		if is_training:
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
			train_mean = tf.assign(pop_mean,
					pop_mean * decay + batch_mean * (1 - decay))
			train_var = tf.assign(pop_var,
					pop_var * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs,
					batch_mean, batch_var, beta, scale, epsilon),batch_mean,batch_var
		else:
			return tf.nn.batch_normalization(inputs,
				pop_mean, pop_var, beta, scale, epsilon),pop_mean,pop_var

	def next_batch(self,data,size = 128,target_num = 10):
		for i in range(size,len(data),size):
			batch_x = np.zeros((size,32,32,1))
			batch_y = np.zeros((size,target_num))
			for j in range(size):
				curr_patch = data[i - size + j][0]
				curr_patch = binary_dilation(curr_patch)
				curr_patch = random_noise(curr_patch,var = .1)
				#print 'max patch',np.max(curr_patch)
				curr_patch = np.float32(curr_patch)
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
				batch_x[j,:,:,0] = curr_patch
				batch_y[j,:] = data[i - size + j][1]
			yield batch_x,batch_y

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		var = tf.Variable(initial)
		weight_decay = tf.multiply(tf.nn.l2_loss(var), 0e-5, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		return var

	def bias_variable(self,shape):
		initial = tf.constant(0., shape=shape)
		return tf.Variable(initial)

	def conv2d(self,x, W,padding = 'SAME'):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)	

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
								strides=[1, 2, 2, 1], padding='SAME')

	def inference(self,target_num = 10):
		if self.trainflag:
			self.x = tf.placeholder(tf.float32,[None,32,32,1])
			self.y_ = tf.placeholder(tf.float32,[None,target_num])
		else:
			self.x = tf.placeholder(tf.float32,[1,None,None,1])
			#self.y_ = tf.placeholder(tf.float32,[None,target_num])
		
		W_conv1 = self.weight_variable([5, 5, 1, 64])
		b_conv1 = self.bias_variable([64])
		tmp_1,_,_ = self.batch_norm_layer(self.conv2d(self.x, W_conv1))
		h_conv1 = tf.nn.relu(tmp_1)
		h_pool1 = self.max_pool_2x2(h_conv1)		

		W_conv2 = self.weight_variable([5, 5, 64, 64])
		b_conv2 = self.bias_variable([64])		

		tmp_2,_,_ = self.batch_norm_layer(self.conv2d(h_pool1, W_conv2))
		h_conv2 = tf.nn.relu(tmp_2)
		h_pool2 = self.max_pool_2x2(h_conv2)		

		W_conv3 = self.weight_variable([5, 5, 64, 64])
		b_conv3 = self.bias_variable([64])		

		tmp_3,_,_ = self.batch_norm_layer(self.conv2d(h_pool2, W_conv3))
		h_conv3 = tf.nn.relu(tmp_3)
		h_pool3 = self.max_pool_2x2(h_conv3)		
		"""
		W_fc1 = self.weight_variable([4 * 4 * 64, 1024])
		b_fc1 = self.bias_variable([1024])		
		"""
		W_fc1 = self.weight_variable([4,4,64,1024])
		b_fc1 = self.bias_variable([1024])
		#h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*64])		

		tmp_fc1,_,_ = self.batch_norm_layer(self.conv2d(h_pool3, W_fc1,'VALID'))
		h_fc1 = tf.nn.relu(tmp_fc1)		
		#h_fc1 = tf.reshape(h_fc1,[-1,1024])
		self.keep_prob = tf.placeholder(tf.float32)
		#h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)		

		W_fc2 = self.weight_variable([1,1,1024, target_num])
		b_fc2 = self.bias_variable([target_num])		
		tmp_fc2,_,_ = self.batch_norm_layer(self.conv2d(h_fc1, W_fc2,'VALID'))

		if self.trainflag:
			h_fc2 = tf.reshape(tmp_fc2,[-1,target_num])
			self.y_conv=tf.nn.softmax(h_fc2)
			self.y_res = tf.argmax(self.y_conv,1)
			self.W_conv1 = W_conv1
		else:
			self.h_fc2 = tmp_fc2
	def train(self, data_set_path, out_path = 'model_tt.ckpt',target_num = 10):
		cross_entropy_mean = -tf.reduce_mean(self.y_ * tf.log(self.y_conv))
		tf.add_to_collection('losses', cross_entropy_mean)
		cross_entropy = tf.add_n(tf.get_collection('losses'), name='total_loss')
		self.l_rate = tf.placeholder(tf.float32)
		train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		sess = self.sess
		sess.run(tf.global_variables_initializer())
		with open(data_set_path,'rb') as f:
			data = pickle.load(f)
		random.shuffle(data)
		for batch_x,batch_y in self.next_batch(data,size = 256):
			valid_x,valid_y = batch_x,batch_y
			print(np.argmax(valid_y,axis = 1))
			break
		train_data = data[256:]
		learn_rate = 1e-3
		phist = .5
		for epic in range(10):
			i=0
			random.shuffle(train_data)
			for batch_x, batch_y in self.next_batch(train_data):			
				if i%10 == 0:
					hit = np.zeros(target_num)
					precision = np.zeros(target_num)
					recall = np.zeros(target_num)
					train_accuracy = accuracy.eval(feed_dict={
						self.x:valid_x/255.0, self.y_: valid_y, self.keep_prob: 1.0, self.l_rate: learn_rate})
					results = self.y_res.eval(feed_dict={
						self.x:valid_x/255.0, self.y_: valid_y, self.keep_prob: 1.0, self.l_rate: learn_rate})
					for v_idx in range(256):
						if valid_y[v_idx,results[v_idx]]:
							hit[results[v_idx]] += 1.0
						precision[results[v_idx]] += 1.0
						recall += valid_y[v_idx,:]
					print(hit/precision)
					print(hit/recall)
					#print()
					print("step %d, training accuracy %g max %g lr %g"%(i, train_accuracy,np.max(valid_x),learn_rate))
					if np.abs(phist - train_accuracy) / phist < .1 :
						learn_rate /= 1.0
					phist = train_accuracy
				train_step.run(feed_dict={self.x: batch_x/255.0, self.y_: batch_y, self.keep_prob: 1.0, self.l_rate: learn_rate	})
				i+=1
		print valid_x.shape
		for i in range(10):
			misc.imsave('valid'+str(i)+'.png',valid_x[i,:,:,0])
		save_path = self.saver.save(sess, out_path)
  		print("Model saved in file: %s" % save_path)
  	def predict(self,image,target_num = 101):
  		res = self.sess.run(self.y_conv,feed_dict={self.x: np.reshape(image,(1,32,32,1))/255.0, self.y_: np.zeros((1,target_num)), self.keep_prob: 1.0})
  		sorted_id = np.argsort(res[0])
  		sorted_id = sorted_id[::-1]
  		#print([self.symbol_list[sorted_id[idx]] for idx in range(5)],[res[0,sorted_id[idx]] for idx in range(5)])
  		return [self.symbol_list[sorted_id[idx]] for idx in range(5)],[res[0,sorted_id[idx]] for idx in range(5)]
  	def test(self,image,target_num = 101):
  		sx,sy = image.shape
  		res = self.sess.run(self.h_fc2,feed_dict={self.x: np.reshape(image,(1,sx,sy,1))/255.0, self.keep_prob: 1.0})
  		m = np.argmax(res,axis=3)
  		m = np.squeeze(m)
  		sx,sy = m.shape
  		rm =  np.max(res,axis=3)
  		rm = np.squeeze(rm)
  		s = [[None for y in range(sy)] for x in range(sx)]
  		print(m.shape)
  		for x in range(sx):
  			for y in range(sy):
	  			if rm[x,y] > 8.:
	  				#print m[x,y]
  					s[x][y]=m[x,y]
  				else:
  					s[x][y] = ''
  		#s = [self.symbol_list[m[i]] * (rm[i]>8.) for i in range(len(m))]
		return s,rm
if __name__ == '__main__':
	from sys import argv
	from scipy import misc
	if argv[1] == 'train':
		with tf.Session() as sess:
			sr = SymbolRecognition(sess)
			sr.train(argv[3],argv[2])
			w1 = sr.W_conv1.eval(session = sess)
			with open('weights.data','wb') as fo:
				pickle.dump(w1,fo)
	else:
		with tf.Session() as sess:
			print 'model_path,',argv[2]
			sr = SymbolRecognition(sess,'/home/zhihaozh/python2_proj/multi/'+argv[2],trainflag = False)
			#sr.train('symbols.data')
			#with open('symbols2.data','rb') as f:
			#	data = pickle.load(f)
			#print(data[0][1])
			image = misc.imread(argv[3])
			s,rm = sr.test(image)
			misc.imsave('confidence.png',rm)
			print s
			#print res
			#print(syms)
			#print(probs)
			#w1 = sr.W_conv1.eval(session = sess)
			#with open('weights.data','wb') as fo:
			#	pickle.dump(w1,fo)