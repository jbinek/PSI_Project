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

print(X.loc[:, 'ECG'])
# print("X")
# print(X)

y = pandas.read_csv(file_list_training[0], header=2, delimiter=",",
                    names=["ANS"], usecols=[2], engine='python')

for i in range(1, len(file_list_training)):
    y = y.append(pandas.read_csv(file_list_training[i], header=2, delimiter=",",
                                 names=["ANS"], usecols=[2], engine='python'))

y = ravel(y)
# print("y")
# print(y)


##########     NEURAL NET     ##########
# Creating neural net using Sckit-learn
classsifier = MLPClassifier(solver='adam', alpha=1e-3, max_iter=1000) #change later to adam
classsifier.fit(X, y)

prediction_file = glob.glob("./test/" + '*.csv')

test_X = pandas.read_csv(prediction_file[0], header=2, delimiter=",",
                    names=["ECG", "EDA"], usecols=[3, 4], engine='python')

test_y = pandas.read_csv(prediction_file[0], header=2, delimiter=",",
                    names=["ANS"], usecols=[2], engine='python')

print("test_x")
print(test_X)

test_y = ravel(test_y)
print("test_y")
print(test_y)

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
figure = plt.figure()
h = .02     #step size for mesh

x_min, x_max = (X.min()-.5).min(), (X.max()+.5).max()
y_min, y_max = (y.min()-.5).min(), (y.max()+.5).max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(3, 1, 1)

ax.set_title("Input data")
# Plot the training points
ax.scatter(X.loc[:, 'ECG'], X.loc[:, 'EDA'], c=y, cmap=cm_bright)
# Plot the testing points
ax.scatter(test_X.loc[:, 'ECG'], test_X.loc[:, 'EDA'], c=test_y, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())


#net
ax = plt.subplot(3, 1, 2)
# Put the result into a color plot
if hasattr(classsifier, "decision_function"):
    Z = classsifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = classsifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
ax.scatter(X.loc[:, 'ECG'], X.loc[:, 'EDA'], c=y, cmap=cm_bright, edgecolors='black', s=25)
# and testing points
ax.scatter(X.loc[:, 'ECG'], X.loc[:, 'EDA'], c=y, cmap=cm_bright, alpha=0.6, edgecolors='black', s=25)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Neural net model")
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')


# BAYES
ax = plt.subplot(3, 1, 3)
# Put the result into a color plot
if hasattr(model, "decision_function"):
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
ax.scatter(X.loc[:, 'ECG'], X.loc[:, 'EDA'], c=y, cmap=cm_bright, edgecolors='black', s=25)
# and testing points
ax.scatter(X.loc[:, 'ECG'], X.loc[:, 'EDA'], c=y, cmap=cm_bright, alpha=0.6, edgecolors='black', s=25)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Naive Bayes model")
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score2).lstrip('0'), size=15, horizontalalignment='right')


figure.subplots_adjust(left=.02, right=.98)
plt.show()