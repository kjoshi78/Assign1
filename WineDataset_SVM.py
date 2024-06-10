#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve, validation_curve
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


# In[2]:


# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 


# In[19]:


df = pd.concat([X, y], axis=1)
df.head()


# In[20]:


y = pd.DataFrame(df['quality'])
df.drop('quality', axis=1)


# In[33]:


y['quality'] = np.where(df['quality'] > 6, 1, 0)


# In[35]:


y['quality'].value_counts()


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=142)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)


# In[27]:


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_val_scaled = sc.fit_transform(X_val)
X_val = pd.DataFrame(X_val_scaled)

X_test_scaled = sc.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


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
    y_subset = y_train.iloc[indices]
    
    subsets.append((X_subset, y_subset))


# In[29]:


# Train SVM models with different kernels
kernels = ['linear', 'rbf', 'poly']
models = []
for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    models.append((model, X_train, X_val, y_train, y_val, kernel))


# In[14]:


def bar_graph(score):
    # Data
    #categories = ['', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA', 'weighted avg']
    #colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'cyan', 'black']
    kernels = ['linear', 'rbf', 'poly']
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(kernels, score, marker='o', linestyle='-', color='b')  # Line plot with markers
    plt.title('Model Performance')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.grid(True)
    plt.tight_layout()

    # Annotate each point
    for i, txt in enumerate(score):
        plt.annotate("{:.4f}".format(txt), (kernels[i], score[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.show()


# In[15]:


# Generate evaluation metrics and plots
weighted_precision_values = []
weighted_recall_values = []
weighted_f1_values = []
for i, (model, X_train, X_val, y_train, y_val, kernel) in enumerate(models):
    y_pred = model.predict(X_test)
    print(f"Kernel: {model.kernel}")
    report = classification_report(y_test, y_pred, output_dict=True)
    weighted_precision_values.append(report['weighted avg']['precision'])
    weighted_recall_values.append(report['weighted avg']['recall'])
    weighted_f1_values.append(report['weighted avg']['f1-score'])
    
print(weighted_precision_values)
print(weighted_recall_values)
print(weighted_f1_values)
bar_graph(weighted_precision_values)
bar_graph(weighted_recall_values)
bar_graph(weighted_f1_values)


# In[16]:


# Learning curves
def plot_learning_curve(estimator, X, y, title, ax, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.legend(loc="best")

fig, axes = plt.subplots(len(models), 1, figsize=(10, 20))

for i, (model, X_train, X_val, y_train, y_val, title) in enumerate(models):
    plot_learning_curve(model, X_train, y_train, f"Learning Curve: {title}, Kernel: {model.kernel}", axes[i])

plt.show()


# In[17]:


param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
}
# Validation curves
def plot_validation_curve(estimator, X, y, param_name, param_range, title, ax, cv=None, n_jobs=None):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(param_range, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(param_range, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    ax.set_title(title)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Score")
    ax.legend(loc="best")

#fig, axes = plt.subplots(4, 2, figsize=(20, 18))
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

(_, X_train, X_val, y_train, y_val, title) = models[1]

#for i, (_, X_train, X_val, y_train, y_val, title) in enumerate(models):

  
plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'gamma', param_grid['gamma'],
                        #f"Validation Curve: {title}, Kernel: RBF, Parameter: gamma", axes[i, 0], cv=5)
                        f"Validation Curve: {title}, Kernel: RBF, Parameter: gamma", axes[0], cv=5)

plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'C', param_grid['C'],
                        f"Validation Curve: {title}, Kernel: RBF, Parameter: C", axes[1], cv=5)
'''
plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'C', param_grid['C'],
                        f"Validation Curve: {title}, Kernel: RBF, Parameter: C", axes[i, 1], cv=5)

plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'gamma', param_grid['gamma'],
                    f"Validation Curve: {title}, Kernel: RBF, Parameter: gamma", axes[i//2, i%2], cv=5)

plot_validation_curve(SVC(kernel='rbf'), X_train, y_train, 'C', param_grid['C'],
                    f"Validation Curve: {title}, Kernel: RBF, Parameter: C", axes[i//2, i%2], cv=5)
'''
plt.show()


# In[34]:


# Define a list of values for C and gamma to try
C_values = [0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1]

best_score = 0
best_params = {'C': None, 'gamma': None}

# Iterate over each combination of C and gamma
for C in C_values:
    for gamma in gamma_values:
        # Create SVC classifier with RBF kernel
        svm = SVC(kernel='rbf', C=C, gamma=gamma)
        # Train the model
        svm.fit(X_train, y_train)
        # Evaluate on the test set
        score = svm.score(X_test, y_test)
        # Check if this combination is the best so far
        
        if score > best_score:
            best_score = score
            best_params['C'] = C
            best_params['gamma'] = gamma
            

# Train the final model with the best parameters
best_model = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print("Best Parameters:", best_params)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[18]:


# Define a list of values for C and gamma to try
C_values = [0.1, 1, 10, 100]

best_score = 0
best_params = {'C': None}

# Iterate over each combination of C and gamma
for C in C_values:
    # Create SVC classifier with RBF kernel
    svm = SVC(kernel='linear', C=C)
    # Train the model
    svm.fit(X_train, y_train)
    # Evaluate on the test set
    score = svm.score(X_test, y_test)
    # Check if this combination is the best so far
    if score > best_score:
        best_score = score
        best_params['C'] = C

# Train the final model with the best parameters
best_model = SVC(kernel='linear', C=best_params['C'])
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
print("Best Parameters:", best_params)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[29]:


histories = list()
for subset in subsets:
    history = model(subset[0], subset[1])
    histories.append(history)


# In[ ]:


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


from sklearn.metrics import accuracy_score
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


# In[62]:


# Create a k-Nearest Neighbors classifier
model = KNeighborsClassifier(n_neighbors=5)

# Create a LearningCurve visualizer
visualizer = LearningCurve(model, scoring='accuracy')

# Fit and visualize the learning curve
visualizer.fit(X_train, y_train)
visualizer.show()

