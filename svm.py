import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

X_train = np.array([[1, 1],[2, 1], [3, 2], [-1, 1], [-2, 1], [-3, 2], [-1, -1], [-2, -3], [-5, -1], [2, -1], [1, -9], [3, -3]])
Y_train = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

classifier = svm.SVC(kernel='linear', gamma=1)
classifier.fit(X_train, Y_train)

X_test = np.array([[2, 2], [-2, 2], [-10, -5], [2, -7], [200, 200], [-999, -999]])
Y_test = np.array([1, 2, 3, 4, 1, 3])


Y_predict = classifier.predict(X_test)

print 'Y_predict: {}'.format(Y_predict)

sk_accuracy = accuracy_score(Y_test, Y_predict)
np_accuracy = np.sum(Y_test == Y_predict) / float(len(Y_test))

print 'sk_accuracy: ', sk_accuracy
print 'np_accuracy: ', np_accuracy