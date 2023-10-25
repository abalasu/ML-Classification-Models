import pandas as pd
import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs

def plot_svm_model(ax, svm_model):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 10)
    yy = np.linspace(ylim[0], ylim[1], 10)

    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm_model.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5,linestyles=['--','-','--'])
    ax.scatter(svm_model.support_vectors_[:,0],svm_model.support_vectors_[:,1],s=50, linewidth=1, facecolors='none')

X, y = make_blobs(n_samples=20, centers=2, random_state=20)
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X,y)

print('SVM Model Support Vectors ', svm_model.decision_function_shape)
print('SVM Model coef ', svm_model.coef_)
plt.scatter(X[:,0],X[:,1],c=y, s=30,cmap=plt.cm.Paired)
ax = plt.gca()
plot_svm_model(ax,svm_model)
plt.show()
X_test = [[]]
