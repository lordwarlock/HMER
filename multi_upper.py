import tensorflow as tf

class MultiUpper(object):
	def __init__(self):
		pass
	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		var = tf.Variable(initial)
		weight_decay = tf.multiply(tf.nn.l2_loss(var), 1e-5, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		return var

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

	def conv2d(self,x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')	

	def bias_variable(self,shape):
		initial = tf.constant(0., shape=shape)
		return tf.Variable(initial)

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
								strides=[1, 2, 2, 1], padding='SAME')

	def creat_upper_layers(self,id,input_layer,input_features_size,image_size):
		with tf.variable_scope('multi_upper_'+str(id)):
			W_conv1 = self.weight_variable([3,3,input_features_size,64])
			b_conv1 = self.bias_variable([64])
			tmp_1,_,_ = self.batch_norm_layer(self.conv2d(input_layer, W_conv1))
			h_conv1 = tf.nn.relu(tmp_1)
			#h_conv1 = tf.nn.relu(self.conv2d(input_layer, W_conv1) + b_conv1)
			h_pool1 = self.max_pool_2x2(h_conv1)

			W_fc1 = self.weight_variable([(image_size * image_size // 16 ) * 64, 128])
			b_fc1 = self.bias_variable([128])		

			h_pool1_flat = tf.reshape(h_pool1, [-1, (image_size * image_size // 16 )*64])
			#tmp_2,_,_ = self.batch_norm_layer(tf.matmul(h_pool1_flat, W_fc1))
			#h_fc1 = tf.nn.relu(tmp_2)
			h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

			W_fc2 = self.weight_variable([128, 2])
			b_fc2 = self.bias_variable([2])

			y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
			y_res = tf.argmax(y_conv,1)

			return y_conv,y_res

		