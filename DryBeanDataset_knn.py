#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install ucimlrepo')
get_ipython().system('pip install scikit-learn')
import os
import warnings

from time import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Import the libs for test split,KNN, svm and DecisionTree Classifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Import the libs for Tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow import feature_column
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import utils 
from yellowbrick.model_selection import LearningCurve


from ucimlrepo import fetch_ucirepo
matplotlib.use('agg')
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore", category=UserWarning)


# In[ ]:


dry_bean = fetch_ucirepo(id=602)
# data (as pandas dataframes) 
X = dry_bean.data.features 
y = dry_bean.data.targets 
# metadata 
print(dry_bean.metadata) 

# variable information 
print(dry_bean.variables)


# In[ ]:


df = pd.concat([X, y], axis=1)
df.head()


# In[ ]:


import seaborn as sns
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)


# In[20]:


# Basic 
feature_columns = df.columns.to_numpy()[:-1]
col = ['blue','green','crimson','black', 'orange','red', 'white']
fig, ax = plt.subplots(4, 4, figsize=(15, 12))
for variable, subplot in zip(feature_columns, ax.flatten()):
    g=sns.histplot(df[variable],bins=30, kde=True, ax=subplot)
    g.lines[0].set_color('crimson')
    g.axvline(x=df[variable].mean(), color='m', label='Mean', linestyle='--', linewidth=2)
plt.tight_layout()

# Some distributions have long tails and most are bi-modal which means that some bean classes should be quite distinct from others.


# In[63]:


## Some distributions have long tails and most are bi-modal which means that some bean classes should be quite distinct from others.


# In[21]:


fig, ax = plt.subplots(8, 2, figsize=(15, 25))
for variable, subplot in zip(feature_columns, ax.flatten()):
    sns.boxplot(x=df['Class'], y= df[variable], ax=subplot, showmeans=True)
plt.tight_layout()
# Good plot to show


# In[22]:


df_filtered = df[feature_columns]
pearson_correlation_matrix = df_filtered.corr("pearson")
plt.figure(figsize=(12,12))
sns.heatmap(pearson_correlation_matrix,vmin=-1, vmax=1,cmap='coolwarm',annot=True, square=True)


# In[24]:


X = df[feature_columns]
y = df['Class']


# In[25]:


# Label Encode for y
lbl = LabelEncoder()
y = lbl.fit_transform(y)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=142)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)


# In[352]:


#X_train.reset_index(drop=True, inplace=True)


# In[27]:


X_train


# In[28]:


fractions = np.arange(.2,.8,.05)
n_samples = len(X_train)
subsets = []

for frac in fractions:
    size = int(frac * n_samples)
    indices = np.random.choice(len(X_train), size, replace=False)    
    X_subset = X_train.iloc[indices]
    '''
    print(indices)
    print(X_train.index)
    print(len(X_train))
    '''
    y_subset = y_train[indices]
    
    subsets.append((X_subset, y_subset))


# In[29]:


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, index=X_train.index)

X_train_partial = []
y_train_partial = []
for subset in subsets:
    X_train_partial_samples, y_train_partial_samples = subset
    X_train_partial_samples_scaled = sc.fit_transform(X_train_partial_samples)
    X_train_partial.append(pd.DataFrame(X_train_partial_samples_scaled))


# In[30]:


from sklearn.metrics import accuracy_score
def train_knnclassifier(algo, X_train, y_train, upper_limit=100, step=2):
    upper_limit = upper_limit
    step = step
    k_range = [i for i in range(2, upper_limit, step)]
    print(f'k_range = {k_range}')
    result = []
    max_k, max_acc = float('-inf'),float('-inf')
    for i in k_range:
        model=KNeighborsClassifier(n_neighbors=i, metric=algo)
        model.fit(X_train,y_train)
        prediction=model.predict(X_val)
        acc = accuracy_score(prediction,y_val)
        #result.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
        print(f"k = {i} and result = {acc}")
        if acc > max_acc:
            max_acc = acc
            max_k = i
        
        result.append(acc)
    
    #print(f"max_k = {max_k} ; max_acc = {max_acc}")
    return result, max_k


# In[34]:


algorithms = ['minkowski', 'cosine']
results = []
val1, k1 = train_knnclassifier(algorithms[0], X_train, y_train)
val2, k2 = train_knnclassifier(algorithms[1], X_train, y_train)
k_range = [i for i in range(2, 10, 2)]

print(f'k_range = {k_range}; val1 = {val1}')
print(f'k_range = {k_range}; val2 = {val2}')
plt.plot(k_range, val1, color='blue',label='minkowski')
plt.plot(k_range, val2, color='green',label='cosine')
plt.legend(loc='upper right')
plt.xticks(k_range)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()


# In[372]:


def get_accuracy(x_train, y_train, k=11):
    model=KNeighborsClassifier(k)
    model.fit(x_train,y_train)
    prediction=model.predict(X_val)
    acc = accuracy_score(prediction,y_val)
    return acc


# In[396]:


acc_partial = []
for subset in subsets:
    X_train, y_train = subset
    acc_partial.append(get_accuracy(X_train, y_train))
plt.plot(fractions, acc_partial, color='blue')
plt.xticks(fractions)
plt.xlabel("Fraction of training")
plt.ylabel("Accuracy")
plt.show()


# In[158]:


print(len(X_train_20_sampled), len(y_train_20_sampled))
print(len(X_train_40_sampled), len(y_train_40_sampled))
print(len(X_train_60_sampled), len(y_train_60_sampled))
print(len(X_train_80_sampled), len(y_train_80_sampled))


# In[175]:


for i in range(len(y_train_sample)):
    print(get_accuracy(x_ss[i], y_train_sample[i]))


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Assuming X_train and y_train are your features and labels for the training data

# Define the parameter grid for k values
param_grid = {'n_neighbors': range(1, 21)}  # Search for k from 1 to 20

for i in range(1, 100, 2):
    # Create a k-Nearest Neighbors classifier
    knn = KNeighborsClassifier()

    # Perform grid search to find the best k value
    grid_search.fit(X_train, y_train)

# Print the best k value and its corresponding accuracy score
print("Best k:", grid_search.best_params_['n_neighbors'])
print("Best accuracy:", grid_search.best_score_)


# In[68]:


from yellowbrick.model_selection import ValidationCurve
#cv = StratifiedKFold(4)
param_range = range(1,500,10)
oz = ValidationCurve(
    KNeighborsClassifier(), param_name="n_neighbors",
    #param_range=param_range, cv=cv, scoring="accuracy", n_jobs=4,
    param_range=param_range, scoring="accuracy",
)
oz.fit(X, y)
oz.show()


# In[63]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Assuming X_train and y_train are your features and labels for the training data

# Define the parameter grid for k values
param_grid = {'n_neighbors': range(1, 21)}  # Search for k from 1 to 20

# Create a k-Nearest Neighbors classifier
knn = KNeighborsClassifier()

# Create a GridSearchCV object
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Perform grid search to find the best k value
grid_search.fit(X_train, y_train)

# Print the best k value and its corresponding accuracy score
print("Best k:", grid_search.best_params_['n_neighbors'])
print("Best accuracy:", grid_search.best_score_)


# In[62]:


# Create a k-Nearest Neighbors classifier
model = KNeighborsClassifier(n_neighbors=5)

# Create a LearningCurve visualizer
visualizer = LearningCurve(model, scoring='accuracy')

# Fit and visualize the learning curve
visualizer.fit(X_train, y_train)
visualizer.show()


# In[ ]:





# In[201]:


param_range = range(1,10302, 100)

plt.plot(param_range,result)
#plt.xticks(param_range)


# In[218]:


print(len(X_val))
print(len(y_val))
from collections import Counter
Counter(y_train)
#Counter(y_val)
prediction


# In[243]:


y_train_1 = to_categorical(y_train)
y_val_1 = to_categorical(y_val)


# In[245]:


utils.set_random_seed(812)
epoch = 500
#CHange the layers or neurons
model = Sequential([
    Dense(units=4, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=5, activation='relu'),
    Dense(units=7, activation='softmax')
    ])
#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
#history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32)
history = model.fit(X_train, y_train_1, epochs=epoch, validation_data=(X_val, y_val_1))



# In[19]:


from sklearn.model_selection import GridSearchCV, train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Create a KNN classifier instance
knn = KNeighborsClassifier()

# Use GridSearchCV
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to your data
grid_search.fit(X_train, y_train)

# Retrieve the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best parameters:", best_params)
print("Best score:", best_score)


# In[ ]:




