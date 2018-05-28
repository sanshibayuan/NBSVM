#coding:utf-8

import numpy as np, pandas as pd 
from keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer   


train_x = open('train_contents.txt',encoding = 'UTF8').read().split('\n')
test_x = open('test_contents.txt',encoding = 'UTF8').read().split('\n')
train_y = to_categorical(open('train_labels.txt',encoding = 'UTF8').read().split('\n'))
test_y = open('test_labels.txt',encoding = 'UTF8').read().split('\n')


count_v0= CountVectorizer();  
counts_all = count_v0.fit_transform(train_x +test_x);
count_v1= CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_train = count_v1.fit_transform(train_x);   
print ("the shape of train is "+repr(counts_train.shape)  )
count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_);  
counts_test = count_v2.fit_transform(test_x);  
print ("the shape of test is "+repr(counts_test.shape)  )
  
tfidftransformer = TfidfTransformer();    
train_x = tfidftransformer.fit(counts_train).transform(counts_train);
test_x = tfidftransformer.fit(counts_test).transform(counts_test); 



class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):#NB
            p = x[y==y_i].sum()
            return (p+1) / ((y==y_i).sum()+1) 
        
        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y))) 
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)#SVM/LR
        return self

preds = np.zeros((4324, 11))
for i in range(1,12):
    print ('fit{}'.format(i))
    nbsvm = NbSvmClassifier()
    nbsvm.fit(train_x,train_y[:,i])  
    pred_test = nbsvm.predict_proba(test_x);  
    preds[:,i-1] = pred_test[:,1]

prd = []
for i in range(0,4324):
    prd.append(np.argmax(preds[i])+1)

count = 0
for i in range(0,4323):
    if int(test_y[i])==prd[i]:
        count+=1

print ('precision_score:' + str(float(count) / len(preds)))





        




