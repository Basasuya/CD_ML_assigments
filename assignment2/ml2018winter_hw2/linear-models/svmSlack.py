import numpy as np
from scipy.optimize import minimize

def f(w, X, y, P, N):
	return 1 / 2 * np.dot(w[1:P+1].T, w[1:P+1]) + 2 * np.dot(w[P+1:].T, w[P+1:]) 

def g(w, X, y, P, N):
	w1 = w[:P+1]
	w2 = w[P+1:]
	
	b = -np.ones((N)) + w2
	A = -y.reshape((N, 1)) * X.T
	return b - np.dot(A, w1)


def h(w, X, y, P, N):
	return w[P+1:]
# def g(w, X, y):
# 	print(X, y, w)
# 	P, N = X.shape
# 	b = -np.ones((N))
# 	A = -y.reshape((N, 1)) * X.T
# 	# tmp1 = np.dot(A, w)
# 	# tmp = b - tmp1
# 	# print("hhh",tmp.shape, tmp1.shape, A.shape, w.shape,b.shape)
# 	return np.reshape(b, (N,1)) - np.dot(A, w)

def svmSlack(X, y):
	'''
	SVM Support vector machine.

	INPUT:  X: training sample features, P-by-N matrix.
		y: training sample labels, 1-by-N row vector.

	OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
		num: number of support vectors

	'''
	
	P, N = X.shape
	w = np.zeros((P + 1 + N, 1))
	X = np.vstack((np.ones((1, X.shape[1])), X))
	result = minimize(f, w, (X, y, P, N), constraints=[dict(type='ineq', fun=g, args=(X, y, P, N)), dict(type='ineq', fun=h, args=(X, y, P, N))], options={'disp':False})
	w = result.x
	dis = np.matmul(X.T, w[:P+1])
# 	print(dis.shape)
	num = np.sum(dis == dis.min()) + np.sum(dis == dis.max())
	return w[:P+1], num