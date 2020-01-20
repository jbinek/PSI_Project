from sklearn.neural_network import MLPClassifier
import glob
import pandas
import numpy as np
from numpy import ravel
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

##########     DATA LOADING     ##########
file_list_training = glob.glob("./training/" + '*.csv')

X = pandas.read_csv(file_list_training[0], header=2, delimiter=",",
                    names=["ECG", "EDA"], usecols=[3, 4], engine='python')

for i in range(1, len(file_list_training)):
    X = X.append(pandas.read_csv(file_list_training[i], header=2, delimiter=",",
                                 names=["ECG", "EDA"], usecols=[3, 4], engine='python'))


y = pandas.read_csv(file_list_training[0], header=2, delimiter=",",
                    names=["ANS"], usecols=[2], engine='python')

for i in range(1, len(file_list_training)):
    y = y.append(pandas.read_csv(file_list_training[i], header=2, delimiter=",",
                                 names=["ANS"], usecols=[2], engine='python'))

y = ravel(y)

prediction_file = glob.glob("./test/" + '*.csv')

test_X = pandas.read_csv(prediction_file[0], header=2, delimiter=",",
                    names=["ECG", "EDA"], usecols=[3, 4], engine='python')

for i in range(1, len(prediction_file)):
    test_X = test_X.append(pandas.read_csv(prediction_file[i], header=2, delimiter=",",
                                 names=["ECG", "EDA"], usecols=[3, 4], engine='python'))

test_y = pandas.read_csv(prediction_file[0], header=2, delimiter=",",
                    names=["ANS"], usecols=[2], engine='python')

for i in range(1, len(prediction_file)):
    test_y = test_y.append(pandas.read_csv(prediction_file[i], header=2, delimiter=",",
                                 names=["ANS"], usecols=[2], engine='python'))

test_y = ravel(test_y)


##########     NEURAL NET     ##########
# Creating neural net using Sckit-learn
classsifier = MLPClassifier(solver='adam', alpha=1e-2, max_iter=1000)
classsifier.fit(X, y)

# Making Predictions
result = classsifier.predict(test_X)

# Getting Accuracy and Statistics
score = classsifier.score(test_X, test_y)
print("Score neural net:", score)
print(metrics.classification_report(test_y, result))
print(metrics.confusion_matrix(test_y, result))


##########     NAIVE BAYES     ##########
# Creating Our Naive Bayes Model Using Sckit-learn
model = GaussianNB()
model.fit(X, y)

# Making Predictions
expected = test_y
predicted = model.predict(test_X)

# Getting Accuracy and Statistics
score2 = model.score(test_X, test_y)
print("Score Naive Bayes:", score2)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# Graphs
h = .02     #step size for mesh

x_min, x_max = (X.min()-.5).min(), (X.max()+.5).max()
y_min, y_max = (y.min()-.5).min(), (y.max()+.5).max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# NEURAL NET
plt.figure(2)
# Put the result into a color plot
if hasattr(classsifier, "decision_function"):
    Z = classsifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = classsifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)


plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("Neural net model")
plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
plt.show()


# BAYES
plt.figure(3)
# Put the result into a color plot
if hasattr(model, "decision_function"):
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("Naive Bayes model")
plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score2).lstrip('0'), size=15, horizontalalignment='right')

plt.show()

