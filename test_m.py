from multi_train import *
from sym2multipatch import *
from sys import argv
mc  = MultiClassData(argv[1])
with tf.Session() as sess:
	mt = MultiTrain(sess,mc,int(argv[2]))
	mt.train(target_num = int(argv[2]))