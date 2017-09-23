import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1], [-5], [-9], [-110], [-999], [-77], [-112], [-432], [-987], [0], [0], [0], [0], [0], [0], [0], [0], [0],
              [1], [2], [3], [4], [5], [9], [555], [99999],[7]]).reshape(27, 1)
Y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

classifier = GaussianNB()
classifier.fit(X, Y)

prediction1 = classifier.predict([[-66], [66], [0], [-100], [49], [42], [-42], [555]])

print 'prdiction1: {}'.format(prediction1)