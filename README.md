# Data_Mining_Exploration
Random learning tasks


# Clustering and Recommender Systems in Data Mining
My implementation of basic data mining techniques, including K-Means clustering with multiple distance metrics, can be found in this repository.
✅ SSE analysis and convergence
✅ User-based and item-based collaborative filtering
✅ Probabilistic matrix factorization (PMF)
✅ Impact analysis of hyperparameters and similarity metrics 


When I started learning about data mining, I wanted to explore:
The internal operation of clustering algorithms
How model behavior is affected by distance metrics
How similarity is calculated in recommender systems
How model comparison is shaped by evaluation metrics (RMSE, MAE, and SSE)
I wanted to create and compare algorithms from scratch rather than treating them like black boxes.




Educational Materials Explored :

The main resources I used for my studies were as follows:
1 Tan, Steinbach, and Kumar's book, Introduction to Data Mining
2 Leskovec, Rajaraman, and Ullman's book, Mining of Massive Datasets (Mining Massive Data Sets), Stanford CS246
Documentation for surprise libraries (for recommender systems)
Studies on matrix factorization and collaborative filtering

I suggest start learning the following if you're new to data mining:

Geometry of distance metrics
Trade-offs between bias and variance in recommender systems
Measures of evaluation that go beyond accuracy




#  Task 1 – K-Means with Multiple Distance Metrics

Implemented **K-Means from scratch** using:

- Euclidean Distance  
- Cosine Distance  
- Jaccard Distance  

###  What Was Evaluated

- Sum of Squared Errors (SSE)  
- Clustering Accuracy (majority label mapping)  
- Convergence speed (iterations & runtime)  
- Different stopping criteria  

### Key Findings

- **Lowest SSE:** Jaccard  
- **Best Accuracy:** Cosine  
- **Fastest Convergence:** Cosine  
- **Weakest Performance:** Euclidean  

> Insight: Lower SSE does not always imply better alignment with ground truth labels.  
> The choice of distance metric fundamentally changes cluster behavior.

---

#  Task 2 – Recommender Systems (MovieLens Dataset)

Compared three recommendation approaches using 5-fold cross-validation:

- **Probabilistic Matrix Factorization (SVD – Surprise)**
- **User-Based Collaborative Filtering**
- **Item-Based Collaborative Filtering**

###  Evaluation Metrics

- RMSE  
- MAE  

---

## 🔹 Model Comparison

| Model | Performance |
|--------|------------|
| PMF (SVD) | Best (Lowest RMSE & MAE) |
| Item-based CF | Moderate |
| User-based CF | Lowest |

 Matrix factorization outperformed neighborhood-based methods due to better handling of sparsity and latent feature learning.

---

##  Similarity Metric Impacts : 

Tested similarity metrics with:

- Cosine  
- MSD (Mean Squared Difference)  
- Pearson  

**Cosine similarity performed best** for both User and Item CF.

probable reason : Cosine works well in high dimensional sparse rating matrices.

---

## Effect of Number of Neighbors (k): 

- Item-based CF improved steadily with larger k  
- User-based CF improved initially, then slightly degraded  
- Optimal k differed for user and item models  

> Hyperparameter tuning must be done separately per model.

---

#  Tech and libraries : 

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- scikit-surprise  

---

#  Key patterns that i saw : 
- Distance metrics reshape clustering geometry.
- Convergence criteria influence final SSE.
- Matrix factorization models outperform memory based methods on sparse data.
- Similarity choice and k-value strongly impact recommender performance.

---

## Future Explrations that can be tried : 

- K-Means++ initialization  
- Silhouette score for validation  
- Regularized matrix factorization  
- Implicit feedback modeling  
- Scaling with Spark MLlib  

---

If you're starting or revising Data Mining, feel free to fork and experiment with different metrics or datasets.
Soon, will share more such tasks.




