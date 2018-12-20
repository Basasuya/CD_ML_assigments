import numpy as np
from scipy.optimize import minimize

def f(w, X, y):
	summ = 0
	for i in range(1, len(w)):
		summ = summ + w[i] * w[i]
	return summ                                                                                                                                                                                                                                                                                                                             
def g(w, X, y):
	print(X, y, w)
	summ = np.zeros((X.shape[1]))
	for i in range(X.shape[1]):
		tmp = -1
		for j in range(X.shape[0]):
			tmp = tmp + X[j][i] * w[j]
#         print(tmp)
		tmp = tmp * y[:, i]
		# print(tmp, y[:, i])
		summ[i] = tmp
	# tmp = b - np.dot(A, w)
	print(summ.shape, summ)
	return summ


def svmHH(X, y):
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
	print(X.shape, y.shape, w.shape)

	g(w, X, y)
	# num = 0
	
	# result = minimize(f, w, (X, y), constraints=[dict(type='ineq', fun=g, args=(X, y))], options={'disp':False})
	# return result.x, result.nit