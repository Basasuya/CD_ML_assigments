import numpy as np
from scipy.optimize import minimize

def f(w, X, y):
	H = np.eye(w.shape[0])
	H[0, 0] = 0
	return 1 / 2 * np.dot(w.T, np.dot(H, w)) 

# def g(w, X, y):
#     P, N = X.shape
#     b = -np.ones((N))
#     A = -y.reshape((N, 1)) * X.T
#     print(b.shape,  np.dot(A, w).shape, (b - np.dot(A, w)).shape)
#     return b - np.dot(A, w)

flag = true;
def g(w, X, y):
	print(X, y, w)
	P, N = X.shape
	b = -np.ones((N))
	A = -y.reshape((N, 1)) * X.T
	tmp1 = np.dot(A, w)
	tmp = b - tmp1
	print("hhh",tmp.shape)
	return tmp

def svm(X, y):
	'''
	SVM Support vector machine.

	INPUT:  X: training sample features, P-by-N matrix.
		y: training sample labels, 1-by-N row vector.

	OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
		num: number of support vectors

	'''
	
	P, N = X.shape
	w = np.zeros((P + 1, 1))
	X = np.vstack((np.ones((1, X.shape[1])), X))
	result = minimize(f, w, (X, y), constraints=[dict(type='ineq', fun=g, args=(X, y))], options={'disp':False})
	w = result.x
	dis = np.matmul(X.T, w)
# 	print(dis.shape)
	num = np.sum(dis == dis.min()) + np.sum(dis == dis.max())
	return w, num