import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[1, 1],[2, 1], [3, 2], [-1, 1], [-2, 1], [-3, 2], [-1, -1], [-2, -3], [-5, -1], [2, -1], [1, -9], [3, -3]])
Y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

classifier = GaussianNB()
classifier.fit(X, Y)

prediction1 = classifier.predict([[2, 2], [-2, 2], [-10, -5], [2, -7]])

print 'prdiction1: {}'.format(prediction1)