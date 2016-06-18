from typing import List

import orchpy
import time
import numpy as np

@orchpy.distributed([tuple], [np.ndarray])
def linreg_gd(XY):
    X = XY[0]
    Y = XY[1]
    alpha = 0.0001
    w = np.zeros((X.shape[1], 1))
    a = X.T.dot(Y)
    b = X.T.dot(X)
    for i in range(75000):
	diff = alpha / (i+1) * 2 * (b.dot(w) - a)
	w -= diff
    return w	

@orchpy.distributed([tuple], [np.ndarray])
def linreg_sgd(XY):
    X = XY[0]
    Y = XY[1]
    alpha = 0.00025
    x = np.copy(X)
    y = np.copy(Y)
    w = np.zeros((X.shape[1],1))
    for i in range(4):
        appended = np.append(x,y,axis=1)
        np.random.shuffle(appended)
        x = appended[:, :-1]
        y = np.array(appended[:, appended.shape[1]-1:])
        prex = x.T.dot(x)
        prey = x.T.dot(y)
        newalpha = alpha / (i+1) 
	for k in range(500):  
	    for j in range(X.shape[1]):
                w[j] -= newalpha * 2 * (prex[j].dot(w) - prey[j])
    return w

def logit(theta, X):
    return 1/ (1+ np.exp(-1*X.dot(theta)))

@orchpy.distributed([tuple], [np.ndarray])
def logreg_gd(XY):
    X = XY[0]
    Y = XY[1]
    alpha = 0.0025
    w = np.zeros((X.shape[1], 1))
    for i in range(1000):
	l = logit(w, X)
        w -= alpha / (i+1) *(l - Y.reshape((len(Y),1))).T.dot(X).reshape((len(w),1))
    return w

@orchpy.distributed([tuple], [np.ndarray])
def logreg_sgd(XY):
    X = XY[0]
    Y = XY[1]
    x = np.copy(X)
    y = np.copy(Y).reshape((X.shape[0],1))
    print x.shape,y.shape
    alpha = 0.0025
    w = np.zeros((X.shape[1], 1))
    for i in range(4):
        appended = np.append(x,y,axis=1)
        np.random.shuffle(appended)
        x = appended[:, :-1]
        y = np.array(appended[:, appended.shape[1] -1:])
        prey = x.T.dot(y)
	for k in range(500):
	    for j in range(X.shape[1]):
		cd = logit(w,x)
		a = x.T[j].dot(cd)
		w[j] -= alpha / (k+1) * (a - prey[j])
    return w

def pred_values(theta, X):
    pred_prob = logit(theta, X)
    pred_value = np.where(pred_prob >= 0.5, 1, 0)
    return pred_value
