import os
import pydot_ng


data_='houston'
SAVE_PTH=os.path.join('../file/'+ data_+'/')
PTH=os.path.join('../data/'+ data_+'/')

os.system('python construct_train_test_mat.py --save_path {}  --PATH {}  --data {} --ksize {}'\
.format(SAVE_PTH,PTH,data_,KSIZE_)) 

regular1_= 0.01
regular2_= 1.

print('NET0')
os.system('python classify.py --mode 0 --NET 0 --data {} --ksize {} --NUM_EPOCH 500 --regular1 {} --regular2 {}'\
.format(data_,KSIZE_,regular1_,regular2_))

print('NET1')
os.system('python classify.py --mode 0 --NET 1 --data {} --ksize {} --NUM_EPOCH 500 --regular1 {} --regular2 {}'\
.format(data_,KSIZE_,regular1_,regular2_))

print('NET2')
os.system('python classify.py --mode 0 --NET 2 --data {} --ksize {} --NUM_EPOCH 500 --regular1 {} --regular2 {}'\
.format(data_,KSIZE_,regular1_,regular2_))

print('NET3')
os.system('python classify.py --mode 0 --NET 3 --data {} --ksize {} --NUM_EPOCH 500 --regular1 {} --regular2 {}'\
.format(data_,KSIZE_,regular1_,regular2_))

print('NET3')
os.system('python classify.py --mode 1 --NET 1 --data {} --ksize {} --NUM_EPOCH 500 --regular1 {} --regular2 {}'\
.format(data_,KSIZE_,regular1_,regular2_))
