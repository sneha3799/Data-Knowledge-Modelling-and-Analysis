#!/usr/bin/env python
# coding: utf-8

# # Assignment 2

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Non Linear Dimensionality Reduction

# ### Local Linear Embedding (LLE)

# In[2]:


dataC = pd.read_csv('DataC.csv')


# In[3]:


dataC.shape


# In[4]:


X = dataC[dataC['gnd']==3]


# In[5]:


X


# In[6]:


X.shape


# In[7]:


from sklearn.manifold import LocallyLinearEmbedding


# In[8]:


import plotly.express as px


# In[9]:


embedding = LocallyLinearEmbedding(n_components=2)


# In[10]:


get_ipython().run_cell_magic('time', '', 'X_transformed = embedding.fit_transform(X)\n')


# In[11]:


X_transformed.shape


# In[12]:


X_transformed


# In[13]:


lle_df = pd.DataFrame(X_transformed, columns=['Feature 1','Feature 2'])
lle_df.head()


# In[14]:


# Plot the 2-D representations
plt.figure(figsize=(10, 8))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], label="LLE 2D Representation")


# In[32]:


# Overlay original images on the plot
for i in range(len(X)):
    plt.text(X_transformed[i, 0], X_transformed[i, 1], str("."), color='red', fontsize=20) # plot dot for each image
plt.title("2-D Representation of Images of Digit '3'")
plt.xlabel("First Component")
plt.ylabel("Second Component")
plt.grid(True)
plt.show()


# ### Isomap

# In[16]:


from sklearn.manifold import Isomap


# In[17]:


embedding_imap = Isomap(n_components=2)


# In[18]:


X


# In[27]:


isomap = embedding_imap.fit(X)


# In[28]:


get_ipython().run_cell_magic('time', '', 'X_transformed_imap = isomap.transform(X)\n')


# In[29]:


# Plot the 2-D representations
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.scatter(X_transformed_imap[:, 0], X_transformed_imap[:, 1], cmap=plt.cm.rainbow, label="Isomap 2D Representation")


# In[30]:


imap_df = pd.DataFrame(X_transformed_imap, columns=['Feature 3','Feature 4'])
imap_df.head()


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


# In[40]:


X = dataC.iloc[:,:-1]
y = dataC.iloc[:,-1]


# In[61]:


from sklearn.metrics import accuracy_score


# In[70]:


iterations = range(5,35,5)
lle_accuracies, imap_accuracies = [], []

for i in iterations:
    lle_accuracy, imap_accuracy = 0,0
    for j in range(i):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
        
        lle = LocallyLinearEmbedding(n_neighbors=i, n_components=4)
        lle_train = lle.fit_transform(X_train)
        lle_test = lle.transform(X_test)
        
        nb_lle = GaussianNB()
        nb_lle.fit(lle_train,y_train)
        y_pred_lle = nb_lle.predict(lle_test)
        lle_accuracy += accuracy_score(y_test,y_pred_lle)
        
        isomap = Isomap(n_neighbors=i, n_components=4)
        imap_train = isomap.fit_transform(X_train)
        imap_test = isomap.transform(X_test)
        
        nb_imap = GaussianNB()
        nb_imap.fit(imap_train,y_train)
        y_pred_imap = nb_imap.predict(imap_test)
        imap_accuracy += accuracy_score(y_test,y_pred_imap)
        
    avg_lle_accuracy = lle_accuracy/i
    avg_imap_accuracy = imap_accuracy/i
    print(avg_lle_accuracy, avg_imap_accuracy)
    
    lle_accuracies.append(avg_lle_accuracy)
    imap_accuracies.append(avg_imap_accuracy)


# In[71]:


sns.lineplot(x=iterations,y=lle_accuracies)
sns.lineplot(x=iterations, y=imap_accuracies)


# ## Binary classification

# In[8]:


dataA = pd.read_csv('DataA1.csv')


# In[9]:


dataA.head()


# In[10]:


dataA.shape


# In[11]:


X = dataA.iloc[:,:-1]
y = dataA.iloc[:,-1]


# In[12]:


from scipy.stats import zscore


# In[13]:


X_scaled = X.apply(zscore)


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state= 42)


# In[16]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ### KNN

# In[17]:


from sklearn.model_selection import cross_val_score


# In[18]:


from sklearn.neighbors import KNeighborsClassifier


# In[19]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# In[20]:


l = range(1,33,2)
accuracy = []
for i in l:
    knn = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(knn, X_train.values,y_train, cv=5, scoring='accuracy')
    accuracy.append(score.mean())


# In[21]:


sns.lineplot(x=l, y=accuracy)


# In[22]:


# model performs best when k is around 6
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)


# In[23]:


y_pred = knn.predict(X_test.values)


# In[24]:


print(accuracy_score(y_test,y_pred))


# ### SVM

# In[25]:


from sklearn.svm import SVC


# In[42]:


cList = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
gList = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
pairs = []
accuracy = []
fList = []

for i in cList:
    for j in gList:
        pairs.append(i+j)
        svm = SVC(C=i,gamma=j,kernel='rbf')
        score = cross_val_score(svm, X_train,y_train, cv=5, scoring='accuracy')
        accuracy.append(score.mean())
        fList.append([[i,j],score.mean()])


# In[43]:


for i,j in fList:
    if j==max(accuracy):
        print(i)


# In[44]:


max(accuracy)


# In[45]:


sns.lineplot(x=pairs,y=accuracy)


# In[46]:


# model performs best when C is around 0.1 and gamma is around 0.01
svm = SVC(C=10,gamma=0.01,kernel='rbf')
svm.fit(X_train,y_train)


# In[47]:


y_pred = svm.predict(X_test)


# In[48]:


print(accuracy_score(y_test,y_pred))


# ### Naive Bayes

# In[49]:


from sklearn.naive_bayes import GaussianNB


# In[50]:


nb = GaussianNB()


# In[51]:


nb.fit(X_train,y_train)


# In[52]:


y_pred = nb.predict(X_test)


# In[53]:


print(accuracy_score(y_test,y_pred))


# ### Decision Tree

# In[54]:


from sklearn.tree import DecisionTreeClassifier


# In[55]:


dtc = DecisionTreeClassifier()


# In[56]:


dtc.fit(X_train,y_train)


# In[57]:


y_pred = dtc.predict(X_test)


# In[58]:


print(accuracy_score(y_test,y_pred))


# In[59]:


from sklearn.model_selection import RepeatedStratifiedKFold


# In[60]:


cv = RepeatedStratifiedKFold(n_repeats=20)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Multiclass classification

# In[64]:


data = pd.read_csv('DataB1.csv')


# In[65]:


data.head()


# In[66]:


data.isnull().sum()


# In[67]:


data.shape


# In[68]:


X = data.iloc[:,:-1]
y = data.iloc[:,-1]


# In[69]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data['class'])
data['class'] = le.transform(data['class'])


# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


from sklearn.preprocessing import StandardScaler


# In[72]:


sc = StandardScaler()
X_scaled = sc.fit_transform(X)


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3,random_state=42)


# In[74]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[75]:


sns.heatmap(data)


# In[76]:


sns.pairplot(data)


# In[77]:


X = data.iloc[:,:-1]
y = data.iloc[:,-1]


# In[78]:


X.shape, y.shape
print(X,y)


# ### SVM

# In[79]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


# In[80]:


from sklearn.svm import SVC


# In[81]:


svc = SVC(decision_function_shape='ovo')


# In[82]:


svc.fit(X_train,y_train)


# In[83]:


y_pred = svc.predict(X_test)


# In[84]:


print(accuracy_score(y_test,y_pred))


# In[85]:


print(confusion_matrix(y_test,y_pred))


# In[86]:


print(precision_score(y_test, y_pred, average=None))


# In[87]:


print(f1_score(y_test,y_pred,average=None))


# In[88]:


print(recall_score(y_test,y_pred,average=None))


# In[89]:


svcr = SVC(decision_function_shape='ovr')


# In[90]:


svcr.fit(X_train,y_train)


# In[91]:


y_pred_r = svcr.predict(X_test)


# In[92]:


print(accuracy_score(y_test,y_pred_r))


# In[93]:


print(confusion_matrix(y_test,y_pred_r))


# In[94]:


print(precision_score(y_test, y_pred_r, average=None))


# In[95]:


print(f1_score(y_test,y_pred_r,average=None))


# In[96]:


print(recall_score(y_test,y_pred_r,average=None))


# ### Decision Tree

# In[97]:


from sklearn.tree import DecisionTreeClassifier


# In[98]:


dtcm = DecisionTreeClassifier()


# In[99]:


dtcm.fit(X_train,y_train)


# In[100]:


y_pred = dtcm.predict(X_test)


# In[101]:


print(accuracy_score(y_test,y_pred))


# In[102]:


print(precision_score(y_test,y_pred,average=None))


# In[103]:


print(recall_score(y_test,y_pred,average=None))


# In[104]:


print(confusion_matrix(y_test,y_pred))


# In[105]:


print(f1_score(y_test,y_pred,average=None))


# In[ ]:




