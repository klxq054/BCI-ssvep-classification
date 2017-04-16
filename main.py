import numpy as np
from multiclass_perceptron import Multiclass_Perceptron
#from perceptron import Perceptron
import csv

with open('Dataset2.csv') as csvfile:
	reader=csv.reader(csvfile)
	data=list(reader)
train_set=[]
test_set=[]
size =len(data)
test_size=size//3
for sample in data[0:test_size]:
	sample=[float(i) for i in sample]
	test_set.append([sample[0:len(sample)-1],int(sample[len(sample)-1])-1])
for sample in data[test_size:size-1]:
	sample=[float(i) for i in sample]
	train_set.append([sample[0:len(sample)-1],int(sample[len(sample)-1])-1])

multi_perc=Multiclass_Perceptron(4020,25,0.001,5,True)
#multi_perc.tuneParameters(train_set,5)
multi_perc.train(train_set)
print("accuracy on the test_set")
multi_perc.test(test_set)
print("accuracy on the train_set")
multi_perc.test(train_set)
"""
perc=Perceptron(784,1,0.01,0)
perc.train(train_set)
#print(perc.w)
print(perc.test(test_set))
"""