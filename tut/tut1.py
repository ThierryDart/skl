# An introduction to machine learning with scikit-learn
# https://scikit-learn.org/stable/tutorial/basic/tutorial.html
#

print("# Handwriting recognition #")
# See https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
print("##### Loading an example dataset #####")
from sklearn import datasets

# Load the iris and digits datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

print("Digits data\n",digits.data)
print("Digits target\n",digits.target)
print("Digits image\n",digits.images[0])

print("##### Learning and predicting #####")
# Support Vector Classification
# For the training set, we’ll use all the images from our dataset, except for the last image, which we’ll reserve for our predicting.

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

# Learn from the model is done by passing our training set to the fit method. 
print("Train:",clf.fit(digits.data[:-1], digits.target[:-1]))

#
print("Predict:",clf.predict(digits.data[-1:]))

print("##### Model persistence #####")
# Save a model in scikit-learn by using Python’s built-in persistence model "pickle"

print("# Iris #")
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
print("Train",clf.fit(X, y))

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print("Predict:",clf2.predict(X[0:1]))
print(y[0])

# In the specific case of scikit-learn, it may be more interesting to use joblib’s replacement for pickle (joblib.dump & joblib.load)
# which is more efficient on big data but it can only pickle to the disk and not to a string
from joblib import dump, load
dump(clf, 'filename.joblib') 
clf = load('filename.joblib') 

print("##### Conventions #####")

print("## Type casting : float64 ##")
# Unless otherwise specified, input will be cast to float64:

import numpy as np
from sklearn import random_projection

# X is float32, which is cast to float64 by fit_transform(X).
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
print(X.dtype)

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)

# Regression targets are cast to float64 and classification targets are maintained:
from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC()

print("Train:",clf.fit(iris.data, iris.target))
print("Predict:",clf.predict(iris.data[:3]))

print("Train:",clf.fit(iris.data, iris.target_names[iris.target]))
print("Predict",clf.predict(iris.data[:3]))


print("## Refitting and updating parameters ##")
# Hyper-parameters of an estimator can be updated after it has been constructed via the set_params() method.
# Calling fit() more than once will overwrite what was learned by any previous fit()

import numpy as n
from sklearn.datasets import load_iris
from sklearn.svm import SVC
X, y = load_iris(return_X_y=True)

clf = SVC()
# The default kernel rbf is first changed to linear
print("Train:",clf.set_params(kernel='linear').fit(X, y))
print("Predict:",(clf.predict(X[:5])))

print("Train:",clf.set_params(kernel='rbf').fit(X, y))
print("Predict:",clf.predict(X[:5]))


print("## Refitting and updating parameters ##")
# When using multiclass classifiers, the learning and prediction task that is performed
# is dependent on the format of the target data fit upon:
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

# the classifier is fit on a 1d array of multiclass labels and the predict() method therefore provides corresponding multiclass prediction
classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X, y).predict(X)

# Possible to fit upon a 2d array of binary label indicators:
y = LabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)


from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)