import pickle
from scipy import misc
import numpy as np

with open('weights.data','rb') as fi:
	indata = pickle.load(fi)

w1 = indata[0]
#w2 = indata[1]
print w1.shape

nw1 = (w1 - np.min(w1)) / (np.max(w1) - np.min(w1))
#for i in range(w1.shape[3]):
#	curr = w1[:,:,0,i]
#	nw1[:,:,0,i] = (curr - np.min(curr)) / (np.max(curr) - np.min(curr))

#nw2 = (w2 - np.min(w2)) / (np.max(w2) - np.min(w2))
img = np.zeros((8*8,8*16))
for i in range(4):
	for j in range(8):
		img[i*8:i*8+5,j*8:j*8+5] = nw1[:,:,0,i*8+j]
"""
img2 = np.zeros((6*64,6*64))
for i in range(64):
	for j in range(64):
		img2[i*6:i*6+5,j*6:j*6+5] = nw2[:,:,i,j]
"""
misc.imsave('w1.png',img)
#misc.imsave('w2.png',img2)