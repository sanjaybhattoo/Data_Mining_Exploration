
import numpy as np
import pandas as pd
import zipfile

def load_data(path='data.zip'):
  with zipfile.ZipFile(path, 'r') as z:
    with z.open('data.csv') as df:
      X_df = pd.read_csv(df, header=None)
    X = X_df.values.astype(np.float64)

    with z.open('label.csv') as df:
      y_df = pd.read_csv(df, header=None)
    y = y_df.values.ravel().astype(int)
  return X,y

X,y = load_data()

X.shape

y.shape

np.unique(y)

print(X)

def euc_dist(a,b):
  diff = a-b
  sqdiff = diff ** 2
  sum_sq = np.sum(sqdiff, axis=1)
  dist = np.sqrt(sum_sq)
  return dist

def cosine_dist(a,b):
  a_norm =np.linalg.norm(a)
  b_norm =np.linalg.norm(b)
  res =np.dot(a,b)/(a_norm * b_norm)
  res =np.clip(res,-1,1)
  return 1-res

def jaccard_dist(a,b):
  a_sum=np.sum(np.minimum(a,b),axis=1)
  m_sum=np.sum(np.maximum(a,b),axis=1)
  res = a_sum/m_sum
  return 1-res

def kmeans(X,K,fun,i=250):
    n,D = X.shape
    centroids = X[np.random.choice(n,K,replace=False)]
    for q in range(i):
        distances = np.zeros((X.shape[0], K))
        for k in range(K):
            distances[:, k] = fun(X, centroids[k])
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            points = X[labels == k]
            if len(points) > 0:
                new_centroids[k] = points.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        centroids = new_centroids
    sse = 0
    for k in range(K):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            sse += np.sum(fun(cluster_points, centroids[k]))
    return labels, centroids, sse
K = len(np.unique(y))

np.random.seed(42)
a1,a2,sse_euc =kmeans(X,K,euc_dist)
a1,a2,sse_cos =kmeans(X,K,cosine_dist)
a1,a2,sse_jac =kmeans(X,K,jaccard_dist)

"""Compare the SSEs of Euclidean-K-means, Cosine-K-means, Jarcard-K-means. Which method is better?"""

print("SSE Euclidean:",sse_euc)
print("SSE Cosine:",sse_cos)
print("SSE Jaccard: ",sse_jac)

ans = min(sse_euc,sse_cos,sse_jac)
if ans ==sse_euc:
    print("\nEuclidean kmeans is better")
elif ans == sse_cos:
    print("\nCosine kmeans is better")
else:
    print("\nJaccard kmeans is better")

"""Q2"""

def compute_accuracy(X, y_true, K,func, i=100):
    N, D = X.shape
    centr = X[np.random.choice(N, K, replace=False)]

    for q in range(i):
        distbw = np.zeros((X.shape[0], K))
        for k in range(K):
            distbw[:, k] = func(X,centr[k])
        labels = np.argmin(distbw,axis=1)

        new_c = np.zeros_like(centr)
        for k in range(K):
            pc=X[labels==k]
            if len(pc)>0:
                new_c[k] =pc.mean(axis=0)
            else:
                new_c[k] = centr[k]

        centr=new_c
    cluster_labels = np.zeros(K, dtype=int)

    for k in range(K):
        mask = (labels == k)
        if np.sum(mask) > 0:
            cluster_labels[k] = np.bincount(y_true[mask]).argmax()
    pred = cluster_labels[labels]
    acc = np.mean(pred == y_true)
    return acc



K = len(np.unique(y))
np.random.seed(42)
acc_euc = compute_accuracy(X,y,K,euc_dist)
acc_cos = compute_accuracy(X,y,K,cosine_dist)
acc_jac = compute_accuracy(X,y,K,jaccard_dist)

print("Kmeans : ")
print("accuracy Euclidean:",acc_euc)
print("accuracy Cosine:",acc_cos)
print("accuracy Jaccard: ",acc_jac)

ans = max(acc_euc,acc_cos,acc_jac)
if ans ==acc_euc:
    print("\nEuclidean kmeans is better")
elif ans ==acc_cos:
    print("\nCosine kmeans is better")
else:
    print("\nJaccard kmeans is better")

"""3."""

import numpy as np
import time

def kmeans_convergence(X, K, func, max_iter=500, no_change=0.0001):
    N, D = X.shape
    centroids = X[np.random.choice(N, K, replace=False)]
    prev_sse = np.inf
    count = 0
    start = time.time()
    for _ in range(max_iter):
        count += 1
        distances = np.zeros((N, K))
        for k in range(K):
            distances[:, k] = func(X, centroids[k])
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            pts = X[labels == k]
            if len(pts) > 0:
                new_centroids[k] = pts.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]
        sse = 0
        for k in range(K):
            pts = X[labels == k]
            if len(pts) > 0:
                sse += np.sum(func(pts, new_centroids[k]))
        if np.all(np.abs(new_centroids - centroids) < no_change):
            break
        if sse > prev_sse:
            break
        centroids = new_centroids
        prev_sse = sse
    elapsed = time.time()-start
    return count, elapsed

K = len(np.unique(y))
np.random.seed(42)
iters_euc, time_euc = kmeans_convergence(X,K,euc_dist)
iters_cos, time_cos = kmeans_convergence(X,K,cosine_dist)
iters_jac, time_jac = kmeans_convergence(X,K,jaccard_dist)

print(f"Euclidean K means: {iters_euc} iterations, {time_euc} sec")
print(f"Cosine K means: {iters_cos} iterations, {time_cos} sec")
print(f"Jaccard K means: {iters_jac} iterations, {time_jac} sec")

print("\n*******Comparison *******")
if iters_euc >= iters_cos and iters_euc >= iters_jac:
    print("Euclidean K means requires the most iterations")
elif iters_cos >= iters_euc and iters_cos >= iters_jac:
    print("Cosine K means requires the most iterations")
else:
    print("Jaccard K means requires the most iterations")

print("\n***** Time Comparison ****")
if time_euc >= time_cos and time_euc >= time_jac:
    print("Euclidean K means takes the most time to converge")
elif time_cos >= time_euc and time_cos >= time_jac:
    print("Cosine K means takes the most time to converge")
else:
    print("Jaccard K means takes the most time to converge")

"""Q4"""

def kmeans_sse_term1(X,K,func,no_change=0.0005):
    N, D = X.shape
    centroids = X[np.random.choice(N, K, replace=False)]
    for _ in range(100):
        distances = np.zeros((X.shape[0], K))
        for k in range(K):
            distances[:, k] = func(X, centroids[k])

        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            points = X[labels == k]
            if len(points) > 0:
                new_centroids[k] = points.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]
        if np.all(np.abs(new_centroids - centroids) < no_change):
            break
        centroids = new_centroids
    sse = 0
    for k in range(K):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            sse += np.sum(func(cluster_points, centroids[k]))
    return sse

def kmeans_sse_term2(X,K,func):
    N,D=X.shape
    centroids = X[np.random.choice(N, K, replace=False)]
    prev_sse = np.inf
    for _ in range(100):
        distances = np.zeros((X.shape[0], K))
        for k in range(K):
            distances[:, k] = func(X, centroids[k])

        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            points = X[labels == k]
            if len(points) > 0:
                new_centroids[k] = points.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]

        sse = 0
        for k in range(K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                sse += np.sum(func(cluster_points, new_centroids[k]))
        if sse > prev_sse:
            break
        prev_sse = sse
        centroids = new_centroids
    return prev_sse

def kmeans_sse_term3(X,K,func,i=150):
    N,D= X.shape
    centroids = X[np.random.choice(N, K, replace=False)]

    for i in range(i):
        distances = np.zeros((N, K))
        for k in range(K):
            distances[:, k] = func(X, centroids[k])
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            points = X[labels == k]
            if len(points) > 0:
                new_centroids[k] = points.mean(axis=0)
            else:
                new_centroids[k] = centroids[k]
        centroids = new_centroids
    sse = 0
    for k in range(K):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            sse += np.sum(func(cluster_points, centroids[k]))

    return sse

K = len(np.unique(y))
np.random.seed(42)
sse_euc1 = kmeans_sse_term1(X,K,euc_dist)
sse_cos1 = kmeans_sse_term1(X,K,cosine_dist)
sse_jac1 = kmeans_sse_term1(X,K,jaccard_dist)
sse_euc2 = kmeans_sse_term2(X,K,euc_dist)
sse_cos2 = kmeans_sse_term2(X,K,cosine_dist)
sse_jac2 = kmeans_sse_term2(X,K,jaccard_dist)
sse_euc3 = kmeans_sse_term3(X,K,euc_dist)
sse_cos3 = kmeans_sse_term3(X,K,cosine_dist)
sse_jac3 = kmeans_sse_term3(X,K,jaccard_dist)

print("process 1: No change in centroid:")
print("Euclidean: ",sse_euc1)
print("Cosine: ",sse_cos1)
print("Jaccard: ",sse_jac1)

print("\nprocess 2: sse increases in next iteration:")
print("Euclidean: ",sse_euc2)
print("Cosine: ",sse_cos2)
print("Jaccard: ",sse_jac2)

print("\nprocess 3: after max iterations: ")
print("Euclidean: ",sse_euc3)
print("Cosine: ",sse_cos3)
print("Jaccard: ",sse_jac3)

"""Q 5.

*   jaccard for k means has given best results with the lowes sse compared to others.
*   cosine and jacarrd also perform better in the accuracies for the label.
*   Cosine is better than jaccard in accuracy.


*   Cosine takes less iteration and time to converge.
*   ecludean is not perfroming well compared to other two methods.
"""



"""# Task 2"""


