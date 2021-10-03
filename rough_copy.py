from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.

def create_class_date():
    weight, y = make_blobs(n_samples=100,
                      n_features=1,
                      centers=2,
                      cluster_std=20,
                      center_box=(40, 160),
                      shuffle=True,
                      random_state=1)
    #print(weight)
    class_data = pd.DataFrame(weight,columns=['Weight'])
    #print(class_data)

    height,y = make_blobs(n_samples=100,
                      n_features=1,
                      centers=2,
                      cluster_std=10,
                      center_box=(100, 200),
                      shuffle=True,
                      random_state=1)

    class_data = pd.concat([class_data,pd.DataFrame(height,columns=['Height'])],axis=1)
    return class_data

import random

print(random.randrange(0,100))
