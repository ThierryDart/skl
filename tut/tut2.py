# Statistical learning: the setting and the estimator object in scikit-learn
# https://scikit-learn.org/stable/tutorial/statistical_inference/settings.html

from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
print(data.shape)

digits = datasets.load_digits()
digits.images.shape

import matplotlib.pyplot as plt 
data = digits.images.reshape((digits.images.shape[0], -1))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r) 
plt.show()



