import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC 
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from numpy import genfromtxt
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


Data=genfromtxt('/Users/SongJialin/Desktop/CS 446/Project/Data_blow_s1.csv', delimiter=',')
example=Data[:,0:4020]
label=Data[:,4020]
x_train, x_test, y_train, y_test = train_test_split(example, label, test_size=0.2,random_state=0)

prediction = dict()
prediction=SVC(C=1, cache_size=500, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=4, gamma='auto', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.000001, verbose=False).fit(x_train, y_train).predict(x_test)


accuracy=accuracy_score(y_test, prediction)

# compute and plot confusion matrix

label=np.arange(1,5,1)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Purples):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label))
    plt.xticks(tick_marks,label)
    plt.yticks(tick_marks,label)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
cm = confusion_matrix(y_test, prediction)
print cm
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm)    

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print cm_normalized
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()


# cross validation

#scores = cross_val_score(prediction,example,label, cv=5)


#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))






