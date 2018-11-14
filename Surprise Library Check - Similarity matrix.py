
# coding: utf-8

# In[1]:


#Libraires used:
import pandas as pd
import numpy as np

#This is a random dataset that needs predictions
data = pd.read_csv("data.csv",index_col = 0)
data.head(10)


# # SURPRISE LIBRARY

# In[2]:


from surprise import Reader, Dataset, trainset
from surprise.prediction_algorithms import KNNWithMeans
from surprise import Reader, Dataset, trainset


# In[3]:


#Build trainset
data['User']=data.index
trainset = pd.melt(data,id_vars='User',value_vars=data.columns.drop('User').tolist()).dropna()
data = data.drop('User',axis=1)
trainset.columns = ['User','Item','Value']


# In[6]:


reader = Reader(rating_scale=(-1000,1000)) #Change to max and min of non-NA data
TRAIN = Dataset.load_from_df(trainset[['User','Item','Value']], reader=reader)
trainset = TRAIN.build_full_trainset()


# In[7]:


sim_options = sim_options={'name':'pearson','min_support':2,'user_based':True}
algo = KNNWithMeans(min_k=2,k=10,sim_options=sim_options)
algo.fit(trainset)


# In[10]:


#Pearson corr of U1 
algo.sim[0]


# # Manual Computation

# In[12]:


from scipy.stats.stats import pearsonr

#We select Person similarity
def pearson_sim(v1, v2, minSimSupport=None):
    #Define common ratings
    indexList = (v1[(~pd.isnull(v1)) & (~pd.isnull(v2))]).index 
    if len(indexList) == 0 or (minSimSupport is not None and len(indexList) < minSimSupport):
        sim = 0
    else:
        sim, pvalue = pearsonr(v1[indexList], v2[indexList])
    return sim, len(indexList)

#Matrix of pearson similarities
sim_df = pd.DataFrame(np.zeros(shape=(10,10)),index=data.index,columns = data.index)

min_support = 2

for i in range(10):
    for j in range(10):
        if i == j:
            sim_df.iloc[i,j] = 1
        else:
            sim_per, count_per =  pearson_sim(data.iloc[i,],data.iloc[j,],2)
            if count_per < min_support:
                sim_df.iloc[i,j] = 0
            else:
                sim_df.iloc[i,j] = sim_per
            
                
print(sim_df.iloc[:,0])

