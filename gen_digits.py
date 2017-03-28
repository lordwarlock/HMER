import pickle
import numpy as np
with open('symbols.txt','r') as fin:
	symbols_dict = dict()
	i=0
	for line in fin:
		symbols_dict[i] = line.strip()
		i+=1

with open('symbols2.data','rb') as fin:
	sd = pickle.load(fin)
digits = [str(i) for i in range(10)]
newdata = []
for image,label in sd:
	l_id = np.argmax(label)
	if symbols_dict[l_id] in digits:
		new_label = np.zeros(10)
		new_label[int(symbols_dict[l_id])] = 1
		newdata.append((image,new_label))

with open('digits.data','wb') as fout:
	pickle.dump(newdata,fout)

