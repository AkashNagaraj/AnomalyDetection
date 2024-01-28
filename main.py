import matplotlib.pyplot as plt
import numpy as np
import random

from scipy.stats import geom
from scipy.stats import dweibull
from scipy.stats import expon
from scipy.stats import hypergeom
from scipy.stats import uniform
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from kmeans_scratch import run_Kmeans


def geometric(size):
    p = 0.65
    data = geom.rvs(p=0.5, size=size)
    return data


def uniform(size):
    start, end = 2,10
    data = uniform.rvs(size=size)
    return data


def weibull(size):
    k = 2.4 # shape
    lam = 5.5 # scale
    data = dweibull.rvs(k, loc=0, scale=lam, size=size)
    return data


def hypergeometric(size):
    [M,n,N] = [20,7,12]
    x = np.random.randint(low=1,high=25,size=(size[0]*size[1]))
    distr = hypergeom(M,n,N)
    data = distr.pmf(x)
    return data


def exponential(size):
    data = expon.rvs(size=size)
    return data


def get_distribution(distr,size):
    match distr:
        case "exponential":
            return exponential(size)
        case "geometric":
            return geometric(size)
        case "uniform":
            return uniform(size)
        case "weibull":
            return weibull(size)
        case "hypergeometric":
            return hypergeometric(size)
        case _:
            return "invalid value"


def default_model(X,y):
    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X)
    pred_classes = kmeans.predict(X) #kmeans.labels_
    centers = kmeans.cluster_centers_
    accuracy = np.count_nonzero(pred_classes==y)/X.shape[0]
    return accuracy


def build_dataset():
    
    rows = [500,1000,1500,2000]#,3000,3500]
    cols = [500,1000,1500,2000]#,2500,3000,3500,4000]
    all_accuracy = []
    
    K = 4
    for row_size in rows:
        for col_size in cols:
            size = (row_size, col_size) #size = (100,5)
            distributions = {"exponential":1,"geometric":2,"weibull":3,"hypergeometric":4}  # uniform
            X = np.concatenate([get_distribution(dist,size).reshape(size[0],-1) for dist,_ in distributions.items()])
            y = np.concatenate([[classes]*size[0] for _,classes in distributions.items()]) 
            #accuracy = default_model(X,y)
            accuracy = run_Kmeans(K, X, y)
            all_accuracy.append(accuracy)
    
    return all_accuracy


def plot_graph(y):
    x = np.arange(len(y))
    plt.plot(x,y)
    plt.show()


def main():
    accuracy = build_dataset()
    print("Avg accuracy :", sum(accuracy)/len(accuracy))
    plot_graph(accuracy)


if __name__=="__main__":
    main()
