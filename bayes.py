# Importing Libraries and Loading Datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import glob
import pandas
import numpy as np
from numpy import ravel
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset = datasets.load_iris()

# Creating Our Naive Bayes Model Using Sckit-learn
model = GaussianNB()
model.fit(dataset.data, dataset.target)

# Making Predictions
expected = dataset.target
predicted = model.predict(dataset.data)

# Getting Accuracy and Statistics
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))



# Graphs
figure = plt.figure(figsize=(17, 9))
h = .02     #step size for mesh

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot()

ax.set_title("Input data")
# Plot the training points
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
# Plot the testing points
ax.scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

# BAYES
ax = plt.subplot()
# Put the result into a color plot
if hasattr(model, "decision_function"):
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='black', s=25)
# and testing points
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, alpha=0.6, edgecolors='black', s=25)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Naive Bayes model")
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score2).lstrip('0'), size=15, horizontalalignment='right')


figure.subplots_adjust(left=.02, right=.98)
plt.show()