import tensorflow as tf
import pickle
import numpy as np
import random
from skimage.morphology import binary_dilation
from skimage.util import random_noise
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data

class SymbolRecognition(object):
	def __init__(self,sess,model_path = None,symbols_path = 'symbols.txt',trainflag = True):
		self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
		self.sess = sess
		self.extra_size = 4
		self.random_brightness = .2
		self.random_br_l = 0.4
		self.random_br_h = 1.2
		self.random_contrast_l = 0.5
		self.random_contrast_h = 1.5
		self.trainflag = trainflag
		self.inference()
		self.saver = tf.train.Saver()
		symbol_list = []
		#with open(symbols_path,'r') as fin:
		#	for line in fin:
		#		symbol_list.append(line.strip())
		#self.symbol_list = symbol_list
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

	def next_batch(self,size = 64,target_num = 10):
		extra_size = self.extra_size
		print(size,target_num)
		for i in range(5000):
			data = self.mnist.train.next_batch(size+2)
			images = np.reshape(data[0],(size+2,28,28))
			labels = data[1]
			batch_x = np.zeros((size,28+extra_size*2,28+extra_size*2,1))
			batch_y = np.zeros((size,target_num+1))
			for j in range(1,size-1):
				prev = images[j-1,:,:]
				next = images[j+1,:,:]
				curr_patch = np.pad(images[j,:,:],\
								((extra_size,extra_size),(extra_size,extra_size)),\
								'constant',constant_values=0)
				if np.random.random() < .1:
					curr_patch[:,:] = 0.
					batch_y[j-1,-1] = 1.
				else:
					batch_y[j-1,:-1] = labels[j,:]
				if np.random.random() >= .2:
					rs = np.random.randint(8,9)
					curr_patch[extra_size:-extra_size,:extra_size+rs] += prev[:,-extra_size-rs:]
				if np.random.random() >= .2:
					rs = np.random.randint(8,9)
					curr_patch[extra_size:-extra_size,-extra_size-rs:] += next[:,:extra_size+rs]
				curr_patch[curr_patch > 1] = 1
				#curr_patch = binary_dilation(curr_patch)
				curr_patch = random_noise(curr_patch,var = .05)
				#print 'max patch',np.max(curr_patch)
				curr_patch = np.float32(curr_patch)
				curr_patch += (np.random.random() - .5) * self.random_brightness
				curr_patch *= np.random.random() * (self.random_br_h - self.random_br_l) + self.random_br_l
				curr_patch[curr_patch > 1] = 1
				curr_patch[curr_patch < 0] = 0
				#contrast adjust
				curr_patch = (curr_patch - np.mean(curr_patch)) \
					* (np.random.random() * (self.random_contrast_h - self.random_contrast_l) + self.random_contrast_l)\
					+ np.mean(curr_patch)
				curr_patch[curr_patch > 1] = 1
				curr_patch[curr_patch < 0] = 0
				batch_x[j-1,:,:,0] = curr_patch
				
			yield batch_x,batch_y
		#print('i=',i)

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		var = tf.Variable(initial)
		weight_decay = tf.multiply(tf.nn.l2_loss(var), 1e-5, name='weight_loss')
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
		extra_size = self.extra_size
		if self.trainflag:
			self.x = tf.placeholder(tf.float32,[None,28+extra_size*2,28+extra_size*2,1])
			self.y_ = tf.placeholder(tf.float32,[None,target_num+1])
			padding = 'VALID'
		else:
			self.x = tf.placeholder(tf.float32,[1,None,None,1])
			padding = 'VALID'
			#self.y_ = tf.placeholder(tf.float32,[None,target_num])
		
		W_conv1 = self.weight_variable([5, 5, 1, 64])
		b_conv1 = self.bias_variable([64])
		tmp_1,_,_ = self.batch_norm_layer(self.conv2d(self.x, W_conv1,padding=padding))
		h_conv1 = tf.nn.relu(tmp_1)
		h_pool1 = h_conv1#self.max_pool_2x2(h_conv1)		

		W_conv2 = self.weight_variable([5, 5, 64, 64])
		b_conv2 = self.bias_variable([64])		

		tmp_2,_,_ = self.batch_norm_layer(self.conv2d(h_pool1, W_conv2,padding=padding))
		h_conv2 = tf.nn.relu(tmp_2)
		h_pool2 = self.max_pool_2x2(h_conv2)		

		W_conv3 = self.weight_variable([5, 5, 64, 64])
		b_conv3 = self.bias_variable([64])		

		tmp_3,_,_ = self.batch_norm_layer(self.conv2d(h_pool2, W_conv3,padding=padding))
		h_conv3 = tf.nn.relu(tmp_3)
		h_pool3 = self.max_pool_2x2(h_conv3)		
		"""
		W_fc1 = self.weight_variable([4 * 4 * 64, 1024])
		b_fc1 = self.bias_variable([1024])		
		"""
		W_fc1 = self.weight_variable([5,5,64,1024])
		b_fc1 = self.bias_variable([1024])
		#h_pool2_flat = tf.reshape(h_pool3, [-1, 4*4*64])		

		tmp_fc1,_,_ = self.batch_norm_layer(self.conv2d(h_pool3, W_fc1,padding))
		h_fc1 = tf.nn.relu(tmp_fc1)	
		self.h_fc1 = h_fc1	
		#h_fc1 = tf.reshape(h_fc1,[-1,1024])
		self.keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)		

		W_fc2 = self.weight_variable([1,1,1024, target_num+1])
		b_fc2 = self.bias_variable([target_num+1])		
		tmp_fc2,_,_ = self.batch_norm_layer(self.conv2d(h_fc1_drop, W_fc2,padding))

		if self.trainflag:
			h_fc2 = tf.reshape(tmp_fc2,[-1,target_num+1])
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
		train_step = tf.train.AdamOptimizer(self.l_rate).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		sess = self.sess
		sess.run(tf.global_variables_initializer())
		#print(self.next_batch(256))
		for batch_x,batch_y in self.next_batch(256):
			valid_x,valid_y = batch_x,batch_y
			break

		learn_rate = 2e-3
		phist = .5
		for epic in range(1):
			i=0
			for batch_x, batch_y in self.next_batch():			
				if i%100 == 0:
					h_fc1 = self.h_fc1.eval(feed_dict={
						self.x:valid_x/1.0, self.y_: valid_y, self.keep_prob: 1.0, self.l_rate: learn_rate})
					print('h_fc1',h_fc1.shape)
					hit = np.zeros(target_num+1)
					precision = np.zeros(target_num+1)
					recall = np.zeros(target_num+1)
					train_accuracy = accuracy.eval(feed_dict={
						self.x:valid_x/1.0, self.y_: valid_y, self.keep_prob: 1.0, self.l_rate: learn_rate})
					results = self.y_res.eval(feed_dict={
						self.x:valid_x/1.0, self.y_: valid_y, self.keep_prob: 1.0, self.l_rate: learn_rate})
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
					if i % 2000 == 0:
						if learn_rate >= 1e-6:
							learn_rate /= 2.
					phist = train_accuracy
				train_step.run(feed_dict={self.x: batch_x/1.0, self.y_: batch_y, self.keep_prob: .5, self.l_rate: learn_rate	})
				i+=1
		print valid_x.shape
		for i in range(20):
			misc.imsave('valid'+str(i)+'_'+str(np.argmax(valid_y[i,:]))+'.png',valid_x[i,:,:,0])
		save_path = self.saver.save(sess, out_path)
  		print("Model saved in file: %s" % save_path)
  	def predict(self,image,target_num = 101):
  		res = self.sess.run(self.y_conv,feed_dict={self.x: np.reshape(image,(1,32,32,1))/1.0, self.y_: np.zeros((1,target_num)), self.keep_prob: 1.0})
  		sorted_id = np.argsort(res[0])
  		sorted_id = sorted_id[::-1]
  		#print([self.symbol_list[sorted_id[idx]] for idx in range(5)],[res[0,sorted_id[idx]] for idx in range(5)])
  		return [self.symbol_list[sorted_id[idx]] for idx in range(5)],[res[0,sorted_id[idx]] for idx in range(5)]
  	def test(self,image,target_num = 101):
  		image = image[4:-4,4:-4]
  		sx,sy = image.shape
  		print(np.max(image))
  		res = self.sess.run(self.h_fc2,feed_dict={self.x: np.reshape(image,(1,sx,sy,1))/255.0, self.keep_prob: 1.0})
  		m = np.argmax(res,axis=3)
  		rm =  np.max(res,axis=3)
  		print m
  		print rm
  		m = np.squeeze(m)
  		rm = np.squeeze(rm)

  		if len(m.shape) == 1:
 			m = np.expand_dims(m,0)
 			print m.shape
	  	sx,sy = m.shape
  		
  		
  		if len(rm.shape) == 1:
 			rm = np.expand_dims(rm,0)
  		s = [[None for y in range(sy)] for x in range(sx)]
  		print(m.shape)
  		for x in range(sx):
  			for y in range(sy):
  				#try:
  				es = self.extra_size
	  			ori_patch = np.array(image[x*4:x*4+36,y*4:y*4+36])
	  			ori_patch[es:-es,es] = 255
	  			ori_patch[es:-es,-es] = 255
	  			ori_patch[es,es:-es] = 255
	  			ori_patch[-es,es:-es] = 255
  				misc.imsave('res/'+'_'.join([str(x),str(y),str(m[x,y]),str(int(rm[x,y]))])+'.png',ori_patch)
  				#except:
  				#	print x,y
  		for x in range(sx):
  			for y in range(sy):
	  			if rm[x,y] > 1.:
	  				#print m[x,y]
  					s[x][y]=m[x,y]
  				else:
  					s[x][y] = m[x,y]
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
			sr = SymbolRecognition(sess,argv[2],trainflag = False)
			#sr.train('symbols.data')
			#with open('symbols2.data','rb') as f:
			#	data = pickle.load(f)
			#print(data[0][1])
			image = misc.imread(argv[3])
			s,rm = sr.test(image)
			misc.imsave('confidence.png',rm)
			print s
			print rm
			#print res
			#print(syms)
			#print(probs)
			#w1 = sr.W_conv1.eval(session = sess)
			#with open('weights.data','wb') as fo:
			#	pickle.dump(w1,fo)