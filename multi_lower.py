import tensorflow as tf
from multi_upper import MultiUpper
class MultiLower(object):
	def __init__(self):
		pass

	def batch_norm_layer(self,inputs, decay = 0.9):
		is_training = True
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

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		var = tf.Variable(initial)
		weight_decay = tf.multiply(tf.nn.l2_loss(var), 1e-5, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		return var

	def bias_variable(self,shape):
		initial = tf.constant(0., shape=shape)
		return tf.Variable(initial)

	def conv2d(self,x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')	

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
								strides=[1, 2, 2, 1], padding='SAME')

	def inference(self,target_num = 101,image_size = 32):
		"""
		NN architecture
		"""
		self.x = tf.placeholder(tf.float32,[None,image_size,image_size])
		self.x_ = tf.expand_dims(self.x,-1)
		self.y_ = tf.placeholder(tf.float32,[None,2])		

		W_conv1 = self.weight_variable([5, 5, 1, 256])
		b_conv1 = self.bias_variable([256])
		#tmp_1,_,_ = self.batch_norm_layer(self.conv2d(self.x, W_conv1))
		#h_conv1 = tf.nn.relu(tmp_1)
		h_conv1 = tf.nn.relu(self.conv2d(self.x_, W_conv1) + b_conv1)
		#h_pool1 = self.max_pool_2x2(h_conv1)		

		W_conv2 = self.weight_variable([5, 5, 256, 256])
		b_conv2 = self.bias_variable([256])
		tmp_2,_,_ = self.batch_norm_layer(self.conv2d(h_conv1, W_conv2))
		h_conv2 = tf.nn.relu(tmp_2)
		#h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)
		self.y_convs = [None for i in range(target_num)]
		self.y_ress = [None for i in range(target_num)]	
		for target_idx in range(target_num):
			curr_y_conv, curr_y_res = MultiUpper().creat_upper_layers(0,h_pool2,256,image_size)
			self.y_convs[target_idx] = curr_y_conv
			self.y_ress[target_idx] = curr_y_res
		#self.y_conv,self.y_res = MultiUpper().creat_upper_layers(0,h_pool2,64)
		return self.y_convs,self.y_ress,self.x,self.y_,W_conv1,W_conv2



if __name__ == '__main__':
	pass