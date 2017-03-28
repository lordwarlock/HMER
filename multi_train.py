import tensorflow as tf
import numpy as np
from multi_lower import MultiLower
from sym2patch import *
import pickle
import random
class MultiTrain(object):
	def __init__(self,sess,data_generator,class_size = 2):
		self.sess = sess
		#self.y_conv,self.y_res,self.x,self.y_,self.W_conv1,self.W_conv2 = MultiLower().inference(2)
		self.y_convs,self.y_ress,self.x,self.y_,self.W_conv1,self.W_conv2 = MultiLower().inference(class_size)
		self.data = data_generator#RealDataBatchGenerator(7,image_path = 'symbols1.data')

	def next_batch(self,size = 64,target_idx = 0):
		#return [(np.random.random((size,32,32)),np.random.random((size,2)))]
		#x,y = self.data.next_batch(size)
		#print 'batch',x.shape,y.shape
		return self.data.next_batch(class_id = target_idx, batch_size = size )

	def train(self, out_path = 'model.ckpt',target_num = 2):
		self.l_rate = tf.placeholder(tf.float32)
		prev_loss = tf.add_n(tf.get_collection('losses'), name='prev_loss')
		cross_entropy_means = [None for i in range(target_num)]
		train_steps = [None for i in range(target_num)]
		correct_predictions = [None for i in range(target_num)]
		accuracys =[None for i in range(target_num)]
		for target_idx in range(target_num):
			y_conv = self.y_convs[target_idx]
			cross_entropy_mean = -tf.reduce_mean(self.y_ * tf.log(y_conv))
			#tf.add_to_collection('losses', cross_entropy_mean)
			#cross_entropy = tf.add_n(tf.get_collection('losses'), name='total_loss')
			cross_entropy = tf.add(prev_loss,cross_entropy_mean,name = str(target_idx) + '_loss')
			train_step = tf.train.AdamOptimizer(self.l_rate).minimize(cross_entropy)
			correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			cross_entropy_means[target_idx] = cross_entropy_mean
			train_steps[target_idx] = train_step
			correct_predictions[target_idx] = correct_prediction
			accuracys[target_idx] = accuracy
		sess = self.sess
		sess.run(tf.initialize_all_variables())
		data = None
		valid_x = [None for i in range(target_num)]
		valid_y = [None for i in range(target_num)]
		for i in range(target_num):
			valid_x[i],valid_y[i] = self.next_batch(size = 256, target_idx = i)
		#print 'valid',valid_x.shape,valid_y.shape
		learn_rate = 1e-5
		phist = .5
		target_idxs = [i for i in range(target_num)]
		for epic in range(5):
			i=0
			for idx in range(10000):
				#print 'learn rate',learn_rate
				#print target_idx
				target_idx = random.choice(target_idxs)
				batch_x, batch_y = self.next_batch(target_idx = target_idx,size = 256)
				#print batch_x.shape,batch_y.shape
				#print np.max(batch_x),np.min(batch_x)
				train_step = train_steps[target_idx]
				correct_prediction = correct_predictions[target_idx]
				accuracy = accuracys[target_idx]
				y_res = self.y_ress[target_idx]
				if i%100 == 0:
					hit = np.zeros(2)
					precision = np.zeros(2)
					recall = np.zeros(2)
					train_accuracy = accuracy.eval(feed_dict={
						self.x:valid_x[target_idx]/255.0, self.y_: valid_y[target_idx], self.l_rate: learn_rate})
					results = y_res.eval(feed_dict={
						self.x:valid_x[target_idx]/255.0, self.y_: valid_y[target_idx], self.l_rate: learn_rate})
					for v_idx in range(256):
						if valid_y[target_idx][v_idx,results[v_idx]]:
							hit[results[v_idx]] += 1.0
						precision[results[v_idx]] += 1.0
						recall += valid_y[target_idx][v_idx,:]
					print(hit/precision)
					print(hit/recall)
					print("step %d, training accuracy %g"%(i, train_accuracy))
					print(target_idx)
					if np.abs(phist - train_accuracy) / phist < .1 :
						learn_rate /= 1.0
					phist = train_accuracy
				train_step.run(feed_dict={self.x: batch_x/255.0, self.y_: batch_y, self.l_rate:learn_rate})
				i+=1
		#save_path = self.saver.save(sess, out_path)
  		#print("Model saved in file: %s" % save_path)
  		print self.W_conv1
  		w1 = self.W_conv1.eval(session = self.sess)
  		w2 = self.W_conv2.eval(session = self.sess)
  		return w1,w2
if __name__ == '__main__':
	with tf.Session() as sess:
		mt = MultiTrain(sess,RealDataBatchGenerator(7,image_path = 'symbols1.data'))
		w1,w2 = mt.train()
		with open('weights.data','wb') as fo:
			pickle.dump([w1,w2],fo)