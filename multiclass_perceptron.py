import numpy as np
from perceptron import Perceptron
from multiprocessing import Pool
class Multiclass_Perceptron:
	def __init__(self,w_size,num_epoch,learning_rate,num_class,decay=False):
		self.num_class=num_class
		self.perceptrons=[Perceptron(w_size,num_epoch,learning_rate,i,decay) for i in range(num_class)]
	
	def train(self,train_data):
		print("Start Training")
		for i in range(len(self.perceptrons)):
			print("Start to train ",i," th"," perceptron")
			self.perceptrons[i].train(train_data)
			print("Finish training ",i," th "," perceptron")

	def test(self,test_data):
		correct=0
		resized_x=[test_data[i][0]+[1] for i in range(len(test_data))]
		y=[test_data[i][1] for i in range(len(test_data))]
		for i in range(len(resized_x)):
			conf=[]
			for perc in self.perceptrons:
				conf.append(perc.predict(resized_x[i]))
			if np.argmax(conf)==y[i]:
				correct+=1
		print(correct/len(test_data))

	def tuneParameters(self,train_set,num_fold):
		print("Start parameter tuning")
		alpha_set=[0.001,0.003,0.01,0.03,0.1]
		folds=[]
		size_fold= len(train_set)//num_fold
		for i in range(num_fold):
			if i!=num_fold-1:
				folds.append(train_set[i:i+size_fold])
			else:
				folds.append(train_set[i:])
		cv_pair=[]
		for i in range(num_fold):
			temp=[]
			for j in range(num_fold):
				if i!=j:
					temp=temp+folds[j]
			cv_pair.append([temp,folds[i]])
		for i in range(self.num_class):
			acc=[]
			for alpha in alpha_set:
				self.perceptrons[i].alpha=alpha
				avg_acc=0 
				for j in range(num_fold):
					self.perceptrons[i].train(cv_pair[j][0])
					avg_acc+=self.perceptrons[i].test(cv_pair[j][1])
					self.perceptrons[i].re_init()
				avg_acc=avg_acc/num_fold
				acc.append(avg_acc)
			self.perceptrons[i].alpha=alpha_set[np.argmax(acc)]
			print("Best alpha for separator of label ",i," is",alpha_set[np.argmax(acc)],"accuracy is",acc[np.argmax(acc)])
		print("Finish parameter tuning")



