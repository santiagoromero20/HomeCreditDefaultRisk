from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""Here the visualization functions called several times will be store, in order to preserve the tidyness of the notebook """

#---------------------------------------------DIMENSIONALITY REDUCTION---------------------------------------------#

def plot_PCA_2D(x,y):

    pca = PCA()
    plt.figure(figsize=(8,6))
    Xt = pca.fit_transform(x)
    plot = plt.scatter(Xt[:,0], Xt[:,1], c=y)
    plt.legend(handles=plot.legend_elements()[0], labels=["Return", "Not Return"])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("First two principal components after scaling")
    plt.show()


def plot_PCA_3D(X, y):

    pca = PCA(n_components=3)
    pca.fit(X) 
    X_pca = pca.transform(X) 

    ex_variance=np.var(X_pca,axis=0)
    ex_variance_ratio = ex_variance/np.sum(ex_variance)
    ex_variance_ratio

    Xax = X_pca[:,0]
    Yax = X_pca[:,1]
    Zax = X_pca[:,2]

    cdict = {0:'purple',1:'yellow'}
    labl = {0:'Return',1:'Non Return'}
    marker = {0:'*',1:'o'}
    alpha = {0:.3, 1:.5}

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor('white')
    for l in np.unique(y):
        ix=np.where(y==l)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
                label=labl[l], marker=marker[l], alpha=alpha[l])
            
    # for loop ends
    ax.set_xlabel("First Principal Component", fontsize=10)
    ax.set_ylabel("Second Principal Component", fontsize=10)
    ax.set_zlabel("Third Principal Component", fontsize=10)

    ax.legend()
    plt.show()


def plot_TSNE(sample_dataset):
    sample_features = sample_dataset.sample(10000)
    sample_class = sample_features.TARGET
    sample_class = sample_class[:,np.newaxis]
    sample_features = sample_features.drop('TARGET',axis=1)

    print("Sample Feature Shape:",sample_features.shape,"Sample Class Shape:", sample_class.shape)

    model = TSNE(n_components=2,random_state=0,perplexity=35)
    t0 = time()
    embedded_data = model.fit_transform(sample_features)
    print("TSNE done in %0.3fs." % (time() - t0))

    final_data = np.concatenate((embedded_data,sample_class),axis=1)
    print(final_data.shape)

    newdf = pd.DataFrame(data=final_data,columns=["Dim1","Dim2","TARGET"])
    sns.FacetGrid(newdf,hue="TARGET",size=6).map(plt.scatter,"Dim1","Dim2").add_legend()
    plt.title("Perplexity=35 with normalization")
    plt.show()


#---------------------------------------------MODEL EVALUATION---------------------------------------------#


def plot_learning_curve(train_sizes, train_scores, validation_scores, ylabel, xlabel, title):

    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, color="blue",label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, color="green",label = 'Validation error')
    plt.ylabel(str(ylabel), fontsize = 14)
    plt.xlabel(str(xlabel), fontsize = 14)
    plt.title(str(title), fontsize = 18, y = 1.03)
    plt.legend()