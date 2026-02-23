!pip install numpy==1.26.4
!pip install scikit-surprise

import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader
from surprise import SVD, KNNBasic
from surprise.model_selection import cross_validate, GridSearchCV
from surprise.accuracy import rmse, mae
sns.set_style('whitegrid')

import numpy as np
import pandas as pd

df1=pd.read_csv('ratings_small.csv')

df1.head()

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1,5))

data = Dataset.load_from_df(df1[['userId','movieId','rating']], reader)

"""#c. Compute the average MAE and RMSE of the Probabilistic Matrix Factorization (PMF), User based Collaborative Filtering, Item based Collaborative Filtering,under the 5-folds cross-validation (10 points)

**the average MAE and RMSE of the Probabilistic Matrix Factorization (PMF)**
"""

pmf=SVD()
pmf_res=cross_validate(pmf,data,measures=['RMSE','MAE'],cv=5,verbose=False)

print(pmf_res)

print("pmf average RMSE:",pmf_res['test_rmse'].mean())
print("pmf average MAE:",pmf_res['test_mae'].mean())

"""**User based Collaborative Filtering**"""

user_based_cf=KNNBasic(sim_options={'name':'msd','user_based':True})
user_based_res=cross_validate(user_based_cf,data,measures=['RMSE','MAE'],cv=5,verbose=False)

print("user based cf average RMSE:", user_based_res['test_rmse'].mean())
print("user based cf average MAE :", user_based_res['test_mae'].mean())



"""**Item based Collaborative Filtering**"""

item_based_cf =KNNBasic(sim_options={'name':'msd','user_based':False})
item_based_res =cross_validate(item_based_cf,data, measures=['RMSE','MAE'],cv=5,verbose=False)

print("item based cf average RMSE:",item_based_res['test_rmse'].mean())
print("item based cf average MAE:",item_based_res['test_mae'].mean())

"""##d. Compare the average (mean) performances of User-based collaborative filtering, item-based collaborative filtering, PMF with respect to RMSE and MAE. Which ML model is the best in the movie rating data? (10 points)"""

print("pmf")
print(pmf_res['test_rmse'].mean())
print(pmf_res['test_mae'].mean())
print("\nuser based cf")
print(user_based_res['test_rmse'].mean())
print(user_based_res['test_mae'].mean())
print("\nitem based cf")
print(item_based_res['test_rmse'].mean())
print(item_based_res['test_mae'].mean())

"""Which ML model is the best in the movie rating data?

Pmf has the lowest rmse and mae error as compared to other two methods which means that it is a good option for the prediction model .also we can see that user based cf is having highest rmse and mae which shows that it has poor similarity so it is not a good option for our ML model.
"""



"""##e. Examine how the cosine, MSD (Mean Squared Difference), and Pearson similarities impact the performances of User based Collaborative Filtering and Item based Collaborative Filtering. Plot your results. Is the impact of the three metrics on User based Collaborative Filtering consistent with the impact of the three metrics on Item based Collaborative Filtering? (10 points)"""

user_e_rmse = []
item_e_rmse = []
user_e_mae = []
item_e_mae = []

user_based1 =KNNBasic(sim_options={'name':'cosine','user_based':True}, k=50)
item_based1 =KNNBasic(sim_options={'name':'cosine','user_based':False}, k=50)
u_res1 =cross_validate(user_based1,data,measures=['RMSE','MAE'],cv=5,verbose=False)
i_res1 =cross_validate(item_based1,data,measures=['RMSE','MAE'],cv=5,verbose=False)
user_e_rmse+=[np.mean(u_res1['test_rmse'])]
item_e_rmse+=[np.mean(i_res1['test_rmse'])]
user_e_mae+=[np.mean(u_res1['test_mae'])]
item_e_mae+=[np.mean(i_res1['test_mae'])]

print("user cosine rmse:",np.mean(u_res1['test_rmse']))
print("item cosine rmse:",np.mean(i_res1['test_rmse']))
print("user cosine mae:",np.mean(u_res1['test_mae']))
print("item cosine mae:",np.mean(i_res1['test_mae']))

user_based2 =KNNBasic(sim_options={'name':'msd','user_based':True},k=50)
item_based2 =KNNBasic(sim_options={'name':'msd','user_based':False}, k=50)
u_res2 =cross_validate(user_based2,data,measures=['RMSE','MAE'],cv=5,verbose=False)
i_res2 =cross_validate(item_based2,data,measures=['RMSE','MAE'],cv=5,verbose=False)
user_e_rmse+=[np.mean(u_res2['test_rmse'])]
item_e_rmse +=[np.mean(i_res2['test_rmse'])]
user_e_mae+=[np.mean(u_res2['test_mae'])]
item_e_mae+=[np.mean(i_res2['test_mae'])]

print("user msd rmse:",np.mean(u_res2['test_rmse']))
print("user msd mae:",np.mean(u_res2['test_mae']))
print("item msd rmse:",np.mean(i_res2['test_rmse']))
print("item msd mae:",np.mean(i_res2['test_mae']))

user_based3 =KNNBasic(sim_options={'name':'pearson','user_based':True},k=50)
item_based3 =KNNBasic(sim_options={'name':'pearson','user_based':False},k=50)
u_res3 =cross_validate(user_based3,data, measures=['RMSE','MAE'], cv=5,verbose=False)
i_res3 =cross_validate(item_based3,data, measures=['RMSE','MAE'],cv=5,verbose=False)
user_e_rmse+=[np.mean(u_res3['test_rmse'])]
item_e_rmse+=[np.mean(i_res3['test_rmse'])]
user_e_mae+=[np.mean(u_res3['test_mae'])]
item_e_mae+=[np.mean(i_res3['test_mae'])]

print("user pearson rmse:",np.mean(u_res3['test_rmse']))
print("user pearson mae:",np.mean(u_res3['test_mae']))
print("item pearson rmse:",np.mean(i_res3['test_rmse']))
print("item pearson mae:",np.mean(i_res3['test_mae']))

print(user_e_rmse)
print(item_e_rmse)
print(user_e_mae)
print(item_e_mae)

import matplotlib.pyplot as plt
x = np.arange(3)
labels = ['Pearson','Cosine','MSD']
width = 0.25
fig, ax = plt.subplots(ncols=2, figsize=(12,5))
ax[0].bar(x - width/2,user_e_rmse,width,label='User cf',color='skyblue')
ax[0].bar(x + width/2,item_e_rmse,width,label='Item cf',color='pink')
ax[0].set_ylabel('rmse')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].set_ylim(0,1)
ax[0].set_title('comparision of Similarity Metric on RMSE')
ax[1].bar(x - width/2,user_e_mae,width,label='User cf',color='skyblue')
ax[1].bar(x + width/2,item_e_mae,width,label='Item cf',color='pink')
ax[1].set_ylabel('mae')
ax[1].set_ylim(0,1)
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].set_title('comparision of Similarity Metric on MAE')
plt.tight_layout()
plt.show()

"""In RMSE, Cosine is performing better than other two for users and items both.
In MAE also cosine is performing better than other two methods for both users and items . Cosine is significantly better for item in both the cases, for users cosine shows better performance by a small margin.
"""



"""##f. Examine how the number of neighbors impacts the performances of User based Collaborative Filtering and Item based Collaborative Filtering? Plot your results. (10 points)"""

n=[5,10,15,20,30,40,50]
user_f_rmse=[]
item_f_rmse=[]
user_f_mae=[]
item_f_mae=[]

for i in n:
  user_based_f =KNNBasic(sim_options={'name':'msd','user_based':True},k=i)
  item_based_f =KNNBasic(sim_options={'name':'msd','user_based':False},k=i)
  u_res_f =cross_validate(user_based_f,data,measures=['RMSE','MAE'],cv=5,verbose=False)
  i_res_f =cross_validate(item_based_f,data,measures=['RMSE','MAE'],cv=5,verbose=False)
  user_f_rmse+=[np.mean(u_res_f['test_rmse'])]
  item_f_rmse+=[np.mean(i_res_f['test_rmse'])]
  user_f_mae+=[np.mean(u_res_f['test_mae'])]
  item_f_mae+=[np.mean(i_res_f['test_mae'])]

print("user rmse: ",user_f_rmse)
print("item rmse: ",item_f_rmse)
print("user mae: ",user_f_mae)
print("item mae: ",item_f_mae)

plt.figure(figsize=(8,5))
plt.plot(n, user_f_rmse, marker='o', label='user rmse CF')
plt.plot(n, item_f_rmse, marker='s', label='item rmse CF')
plt.plot(n, user_f_mae, marker='o', label='user mae CF')
plt.plot(n, item_f_mae, marker='s', label='item mae CF')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Average RMSE')
plt.title('Effect of Number of Neighbors on RMSE')
plt.legend()
plt.xticks(n)
plt.grid(True)
plt.show()

"""As the no of neighours (n) increases item based cf improves a lot with smaller n then it gets almost constant improvement with increasing n .

Another thing i noticed is that user based cf gets a little worse with n after a sudden improvement in the initial n values which may be due to noise from different neighour users.

g
"""

best_k_user = n[np.argmin(user_f_rmse)]
best_rmse_user = min(user_f_rmse)

best_k_item = n[np.argmin(item_f_rmse)]
best_rmse_item = min(item_f_rmse)

print(f"best n for user based cf is : {best_k_user}, and RMSE for the same is : {best_rmse_user:.4f}")
print(f"best n for user based cf is : {best_k_item}, and RMSE for the same is : {best_rmse_item:.4f}")

"""Is the best K of User based collaborative filtering the same with the best K of Item based collaborative filtering?

No, the K value is not the same for both , for user cf it gets best at 15 for my n samples while for item cf it gets best near 50 which was my last n sample.
"""
