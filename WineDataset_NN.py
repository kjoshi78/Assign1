#!/usr/bin/env python
# coding: utf-8

# In[202]:


get_ipython().system('pip install tensorflow')
get_ipython().system('pip install ucimlrepo')
get_ipython().system('pip install scikit-learn')
import os
import warnings
import keras

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
from sklearn.metrics import accuracy_score

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
from IPython.display import clear_output

matplotlib.use('agg')
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore", category=UserWarning)


# In[126]:


# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 


# In[127]:


df = pd.concat([X, y], axis=1)
df.head()


# In[128]:


y = pd.DataFrame(df['quality'])
df.drop('quality', axis=1)


# In[133]:


y['quality'] = np.where(df['quality'] > 6, 1, 0)
y


# In[134]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=142)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)


# In[190]:


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_val_scaled = sc.fit_transform(X_val)
X_val = pd.DataFrame(X_val_scaled)

X_test_scaled = sc.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# In[138]:


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
    y_subset = y_train.iloc[indices]
    
    subsets.append((X_subset, y_subset))


# In[139]:


for subset in subsets:
    len_x, len_y = len(subset[0]), len(subset[1])
    print(f'len(x) = {len_x} ; len(y) ={len_y}')


# In[140]:


# Create the output dir
output_dir = '/tmp/logs'
os.makedirs(output_dir, exist_ok=True)
os.path.exists(output_dir)


# In[149]:


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []


    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric, color='red')
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


# In[146]:


y_val


# In[150]:


callbacks_list = [PlotLearning()]#, TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 500
model = Sequential([
    Dense(units=4, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=1, activation='sigmoid')
    ])
#Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[174]:


callbacks_list = [PlotLearning()]#, TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 100
model_actual = Sequential([
    Dense(units=22, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=1, activation='sigmoid')
    ])
#Compile the model
model_actual.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model_actual.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[ ]:


callbacks_list = [PlotLearning()]#, TensorBoard(log_dir=output_dir)]
#callbacks_list = [TensorBoard(log_dir=output_dir)]

# Build the model
utils.set_random_seed(812)
epoch = 100
model_actual = Sequential([
    Dense(units=22, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=10, activation='relu', input_shape = (X_train.shape[1],)),
    Dense(units=1, activation='sigmoid')
    ])
#Compile the model
model_actual.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and capture the training history
history = model_actual.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)


# In[ ]:


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


# In[158]:


# model
def model(X_train, y_train):
    callbacks_list = [PlotLearning(), TensorBoard(log_dir=output_dir)]
    #callbacks_list = [TensorBoard(log_dir=output_dir)]

    # Build the model
    utils.set_random_seed(812)
    epoch = 100
    model = Sequential([
        Dense(units=50, activation='relu', input_shape = (X_train.shape[1],)),
        Dense(units=1, activation='sigmoid')
        ])
    #Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model and capture the training history
    #history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32, callbacks=callbacks_list)
    history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_val, y_val), batch_size=32)
    
    return history


# In[159]:


histories = list()
for subset in subsets:
    history = model(subset[0], subset[1])
    histories.append(history)


# In[165]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))


axes[0,0].plot(histories[0].history['loss'],color='red',label='20%')
axes[0,0].plot(histories[1].history['loss'],color='orange',label='40%')
axes[0,0].plot(histories[2].history['loss'],color='magenta',label='60%')
axes[0,0].plot(histories[3].history['loss'],color='blue',label='80%')
axes[0,0].set_xlabel("Percentage of Data")
axes[0,0].set_ylabel("Loss")
axes[0,0].legend(loc="upper right")

axes[0,1].plot(histories[0].history['val_loss'],color='red',label='20%')
axes[0,1].plot(histories[1].history['val_loss'],color='orange',label='40%')
axes[0,1].plot(histories[2].history['val_loss'],color='magenta',label='60%')
axes[0,1].plot(histories[3].history['val_loss'],color='blue',label='80%')
axes[0,1].set_xlabel("Percentage of Data")
axes[0,1].set_ylabel("Validation Loss")
axes[0,1].legend(loc="upper right")

axes[1,0].plot(histories[0].history['accuracy'],color='red',label='20%')
axes[1,0].plot(histories[1].history['accuracy'],color='orange',label='40%')
axes[1,0].plot(histories[2].history['accuracy'],color='magenta',label='60%')
axes[1,0].plot(histories[3].history['accuracy'],color='blue',label='80%')
axes[1,0].set_xlabel("Percentage of Data")
axes[1,0].set_ylabel("Accuracy")
axes[1,0].legend(loc="upper right")

axes[1,1].plot(histories[0].history['accuracy'],color='red',label='20%')
axes[1,1].plot(histories[1].history['accuracy'],color='orange',label='40%')
axes[1,1].plot(histories[2].history['accuracy'],color='magenta',label='60%')
axes[1,1].plot(histories[3].history['accuracy'],color='blue',label='80%')
axes[1,1].set_xlabel("Percentage of Data")
axes[1,1].set_ylabel("Validation Accuracy")
axes[1,1].legend(loc="upper right")


# In[182]:


accuracy = model_actual.evaluate(X_test, y_test)[1]
print(accuracy)


# In[191]:


X_test


# In[197]:


from sklearn.metrics import classification_report
y_pred = model_actual.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
print(y_pred_class)
print(classification_report(y_test, y_pred_class))


# In[201]:


from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)
print(conf_matrix)


# In[171]:


def get_accuracy(x_train, y_train):
    model_obj=model(x_train, y_train)
    model_obj.fit(x_train,y_train)
    prediction=model.predict(X_val)
    acc = accuracy_score(prediction,y_val)
    return acc


# In[172]:


acc_partial = []
for subset in subsets:
    X_train, y_train = subset
    acc_partial.append(get_accuracy(X_train, y_train))
plt.plot(fractions, acc_partial, color='blue')
#plt.xticks(k_range)
plt.xlabel("Fraction of training")
plt.ylabel("Accuracy")


# In[53]:


# Basic 
feature_columns = df.columns.to_numpy()[:-1]
# remove the ones which have >.97 correlation.
'''
col = ['blue','green','crimson','black', 'orange','red', 'white']
fig, ax = plt.subplots(4, 4, figsize=(15, 12))
for variable, subplot in zip(feature_columns, ax.flatten()):
    for i,v in enumerate(['DERMASON','SIRA,SEKER' ,'HOROZ','CALI','BARBUNYA' ,'BOMBAY']):
        df1 = df[df['Class']==v]
        g=sns.histplot(df1[variable],bins=30, kde=True, ax=subplot)
        g.lines[0].set_color(col[i])
        #g.lines[0].set_color('crimson')
        g.axvline(x=df1[variable].mean(), color='m', label='Mean', linestyle='--', linewidth=2)
plt.tight_layout()
'''
#col = ['blue','green','crimson','black', 'orange','red', 'white']
col = ['blue','green','crimson','black', 'orange','red', 'white']
fig, ax = plt.subplots(4, 4, figsize=(15, 12))
for variable, subplot in zip(feature_columns, ax.flatten()):
    g=sns.histplot(df[variable],bins=30, kde=True, ax=subplot)
    g.lines[0].set_color('crimson')
    g.axvline(x=df[variable].mean(), color='m', label='Mean', linestyle='--', linewidth=2)
    '''
    for i,v in enumerate(['DERMASON','SIRA,SEKER' ,'HOROZ','CALI','BARBUNYA' ,'BOMBAY']):
        df1 = df[df['Class']==v]
        g.axvline(x=df1[variable].mean(), color=col[i], label='Mean', linestyle='--', linewidth=2)
    '''
plt.tight_layout()

# Some distributions have long tails and most are bi-modal which means that some bean classes should be quite distinct from others.


# In[63]:


## Some distributions have long tails and most are bi-modal which means that some bean classes should be quite distinct from others.


# In[54]:


fig, ax = plt.subplots(8, 2, figsize=(15, 25))
for variable, subplot in zip(feature_columns, ax.flatten()):
    sns.boxplot(x=df['Class'], y= df[variable], ax=subplot, showmeans=True)
plt.tight_layout()
# Good plot to show


# In[24]:


'''
fig, ax = plt.subplots(4, 4, figsize=(15, 12))

for variable, subplot in zip(feature_columns, ax.flatten()):
    sns.boxplot(y= df[variable], ax=subplot)
plt.tight_layout()
'''
# Not needed.


# In[55]:


df_filtered = df[feature_columns]
pearson_correlation_matrix = df_filtered.corr("pearson")
plt.figure(figsize=(12,12))
sns.heatmap(pearson_correlation_matrix,vmin=-1, vmax=1,cmap='coolwarm',annot=True, square=True)
# Can show this data
# Different type of training data - plot the acuuracies with all features and remove correlerated features.(100%)
# 20 % -  plot the acuuracies with all features and remove correlerated features.(100%)
# 40 % -  plot the acuuracies with all features and remove correlerated features.(100%)
# 80 % -  plot the acuuracies with all features and remove correlerated features.(100%)


# In[56]:


# Remove highly correlated features
threshold = 0.8
corr_columns = df_filtered.columns
drop_columns = set()

for i in range(len(corr_columns)):
    for j in range(i):
        if pearson_correlation_matrix.iloc[i, j] >= threshold:
            colname = corr_columns[i]
            drop_columns.add(colname)
print(drop_columns)
# Drop the highly correlated features


# In[154]:


'''
# Remove the outliers
def remove_outliers(beans_data, column):
    Q1 = beans_data[column].quantile(0.25)
    Q3 = beans_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    #beans_data.drop(beans_data[(beans_data[column] < lower_bound) | (beans_data[column] > upper_bound)].index, inplace=True)
    count_df = beans_data[(beans_data[column] < lower_bound) | (beans_data[column] > upper_bound)]
    print(f'Col = {column} = {count_df.shape}')
    
# Apply remove_outliers function to each column
for col in feature_columns:
    remove_outliers(df, col)
'''


# In[349]:


X = df[feature_columns]
y = df['Class']


# In[350]:


# Label Encode for y
lbl = LabelEncoder()
y = lbl.fit_transform(y)


# In[351]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=142)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)

'''
# Randomly Sample 20, 40, 60, 80 of the X_train and X_test
X_train_20_sampled, _, y_train_20_sampled, _ = train_test_split(X_train, y_train, train_size=0.2, random_state=42)
X_train_40_sampled, _, y_train_40_sampled, _ = train_test_split(X_train, y_train, train_size=0.4, random_state=42)
X_train_60_sampled, _, y_train_60_sampled, _ = train_test_split(X_train, y_train, train_size=0.6, random_state=42)
X_train_80_sampled, _, y_train_80_sampled, _ = train_test_split(X_train, y_train, train_size=0.8, random_state=42)
'''


# In[352]:


X_train.reset_index(drop=True, inplace=True)


# In[341]:


#X_train.reset_index(drop=True)
X_train


# In[370]:


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


# In[371]:


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, index=X_train.index)

X_train_partial = []
y_train_partial = []
for subset in subsets:
    X_train_partial_samples, y_train_partial_samples = subset
    X_train_partial_samples_scaled = sc.fit_transform(X_train_partial_samples)
    X_train_partial.append(pd.DataFrame(X_train_partial_samples_scaled))

'''
X_train_20_sampled_scaled = sc.fit_transform(X_train_20_sampled)
X_train_20_sampled = pd.DataFrame(X_train_20_sampled_scaled, index=X_train_20_sampled.index)

X_train_40_sampled_scaled = sc.fit_transform(X_train_40_sampled)
X_train_40_sampled = pd.DataFrame(X_train_40_sampled_scaled, index=X_train_40_sampled.index)

X_train_60_sampled_scaled = sc.fit_transform(X_train_60_sampled)
X_train_60_sampled = pd.DataFrame(X_train_60_sampled_scaled, index=X_train_60_sampled.index)

X_train_80_sampled_scaled = sc.fit_transform(X_train_80_sampled)
X_train_80_sampled = pd.DataFrame(X_train_80_sampled_scaled, index=X_train_80_sampled.index)

X_val_scaled = sc.transform(X_val)
X_val = pd.DataFrame(X_val_scaled, index=X_val.index)
'''


# In[312]:


'''
x_ss = []
sc = StandardScaler()
for x in X_train_sample:
    x_ss.append(pd.DataFrame(sc.fit_transform(x),index=x.index))
'''

    


# In[98]:


y_val


# In[180]:


def train_knnclassifier(algo, X_train, y_train, upper_limit=100, step=2):
    upper_limit = upper_limit
    step = step
    k_range = [i for i in range(2, upper_limit, step)]
    print(f'k_range = {k_range}')
    result = []
    max_k, max_acc = float('-inf'),float('-inf')
    for i in k_range:
        model=KNeighborsClassifier(i, metric=algo)
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


# In[181]:


algorithms = ['minkowski', 'cosine']
results = []
val1, k1 = train_knnclassifier(algorithms[0], X_train, y_train)
val2, k2 = train_knnclassifier(algorithms[1], X_train, y_train)
k_range = [i for i in range(2, 100, 2)]
'''
for algo in algorithms:
    val, idx = train_knnclassifier(algo)
    results.append(val)
print(results[0])
print(results[1])
'''
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



# In[ ]:





# In[ ]:




