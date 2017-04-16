import numpy as np
class Perceptron:
	def __init__(self,w_size,num_epoch,learning_rate,label,decay=False):
		#self.w=np.random.rand(w_size+1)
		#self.w=np.random.normal(size=w_size+1)
		#self.w=np.random.uniform(size=w_size+1)
		self.w=np.zeros(w_size+1)
		self.alpha=learning_rate
		self.epoch=num_epoch
		self.label=label
		self.decay=decay

	def train(self,train_data):
		epoch=0
		while epoch<self.epoch:	
			resized_x=[train_data[i][0]+[1] for i in range(len(train_data))]
			y=[train_data[i][1] for i in range(len(train_data))]
			for i in range(len(resized_x)):
				dot=np.dot(self.w,resized_x[i])
				if y[i]==self.label and np.sign(dot)<=0:
					self.w=np.add(self.w,np.multiply(resized_x[i],self.alpha))
				if y[i]!=self.label and np.sign(dot)>0:
					self.w=np.subtract(self.w,np.multiply(resized_x[i],self.alpha))
			epoch+=1
			if self.decay:
				self.decayAlpha(epoch)
			np.random.shuffle(train_data)

	def test(self,test_data):
		resized_x=[test_data[i][0]+[1] for i in range(len(test_data))]
		y=[test_data[i][1] for i in range(len(test_data))]
		correct=0
		for i in range(len(resized_x)):
			dot=np.dot(self.w,resized_x[i])
			if (np.sign(dot)>0 and self.label==y[i]) or (np.sign(dot)<=0 and self.label!=y[i]):
				correct+=1
		return correct/len(resized_x)

	def predict(self,sample):
		return np.dot(sample,self.w)

	def re_init(self):
		self.w=np.zeros(len(self.w))

	def decayAlpha(self,epoch):
		self.alpha=self.alpha/epoch

