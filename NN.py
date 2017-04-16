
import tensorflow as tf
import numpy as np
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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


feature_columns=[tf.contrib.layers.real_valued_column("",dimension=4020)]
classifier=tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[5000],n_classes=5,activation_fn=tf.nn.relu)
def get_train_set():
	x=tf.constant([train_set[i][0] for i in range(len(train_set))])
	y=tf.constant([train_set[i][1] for i in range(len(train_set))])
	return x,y
def get_test_set():
	x=tf.constant([test_set[i][0] for i in range(len(test_set))])
	y=tf.constant([test_set[i][1] for i in range(len(test_set))])
	return x,y
classifier.fit(input_fn=get_train_set,steps=1000)
accuracy_score=classifier.evaluate(input_fn=get_test_set,steps=1)["accuracy"]
print("\nTest Accuracy:{0:f}\n".format(accuracy_score))
accuracy_score=classifier.evaluate(input_fn=get_train_set,steps=1)["accuracy"]
print("\nTrain Accuracy:{0:f}\n".format(accuracy_score))
